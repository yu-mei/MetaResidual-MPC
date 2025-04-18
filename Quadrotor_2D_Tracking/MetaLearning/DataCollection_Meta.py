import os
import numpy as np
import pandas as pd
import casadi as cs
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import scipy.linalg
import torch
import l4casadi as l4c

np.random.seed(43)
output_dir = "meta_dataset_quadrotor"           # load the Quadrotor_2D_Stabilization as parent folder
os.makedirs(output_dir, exist_ok=True)
m_nominal_ratio = 0.66
Iyy_nominal_ratio = 0.8

# ---------------------------------------------
# Nominal Dynamics (fixed mass and Iyy)
# ---------------------------------------------
class QuadrotorNominalDynamics:
    def __init__(self, env):
        self.env = env

    def model(self):
        m = 0.027 * m_nominal_ratio
        Iyy = 1.4e-5 * Iyy_nominal_ratio
        g = self.env.GRAVITY_ACC
        L = self.env.L

        # State and input
        x, x_dot = cs.MX.sym('x'), cs.MX.sym('x_dot')
        z, z_dot = cs.MX.sym('z'), cs.MX.sym('z_dot')
        theta, theta_dot = cs.MX.sym('theta'), cs.MX.sym('theta_dot')
        T1, T2 = cs.MX.sym('T1'), cs.MX.sym('T2')

        X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        U = cs.vertcat(T1, T2)
        nx, nu = 6, 2
        # Compute the input constraint
        n_mot = 4 / nu
        a_low = self.env.KF * n_mot * (self.env.PWM2RPM_SCALE * self.env.MIN_PWM + self.env.PWM2RPM_CONST)**2
        a_high = self.env.KF * n_mot * (self.env.PWM2RPM_SCALE * self.env.MAX_PWM + self.env.PWM2RPM_CONST)**2
        u_min= a_low * np.ones(nu)
        u_max = a_high * np.ones(nu)

        x_ddot = cs.sin(theta) * (T1 + T2) / m
        z_ddot = cs.cos(theta) * (T1 + T2) / m - g
        theta_ddot = L * (T2 - T1) / Iyy / cs.sqrt(2)

        f_expl = cs.vertcat(x_dot, x_ddot, z_dot, z_ddot, theta_dot, theta_ddot)

        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 6)
        model.u = U
        model.u_min = u_min
        model.u_max = u_max
        model.f_expl = f_expl
        model.name = "quadrotor_nominal"
        model.x_start = np.zeros(6)
        model.p = cs.vertcat([])
        model.z = cs.vertcat([])
        model.constraints = cs.vertcat([])

        return model

# ---------------------------------------------
# MPC Wrapper (simplified for this case)
# ---------------------------------------------
class MPC:
    def __init__(self, model, N, t_horizon):
        self.model = model
        self.N = N
        self.t_horizon = t_horizon

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        
        # Dimensions
        nx = 6
        nu = 2
        ny = nx + nu     # [x, x_dot, z, z_dot, theta, theta_dot, u1, u2]
        ny_e = nx     # Terminal cost considers only state (6)

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # State
        ocp.cost.Vx = np.zeros((ny, nx))
        for i in range(nx):
            ocp.cost.Vx[i, i] = 1 
        # input
        ocp.cost.Vu = np.zeros((ny, nu))
        for i in range(nu):
            ocp.cost.Vu[i + nx, i] = 1
        ocp.cost.Vz = np.array([[]])
        # terminal
        ocp.cost.Vx_e = np.eye(nx)

        l4c_y_expr = None
       
        # Define weight matrices: Q for state and R for control
        Q = 1*np.diag([5, 0.1, 5, 0.1, 0.1, 0.1])
        R = 1*np.diag([0.1, 0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.W_e = Q
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = model.u_max
        a_min = model.u_min
        ocp.constraints.lbu = a_min
        ocp.constraints.ubu = a_max
        ocp.constraints.idxbu = np.arange(nu)

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        return model_ac

# ---------------------------------------------
# Simulation and Data Collection Loop
# ---------------------------------------------
dt = 0.02
T = 15.0
Steps = int(T / dt)
N = 20
t_horizon = 1.0
episodes_per_task = 1
mass_ratios = np.round(np.linspace(0.75, 2.0, 10), 3)
Iyy_ratios = np.round(np.linspace(0.75, 2.0, 5), 3)
targets = [(1.0, 1.0), (0.0, 1.5), (-1.0, 1.0), (0.0, 0.5)]

all_data = []
task_id = 0

for ratio1 in mass_ratios:
    for ratio2 in Iyy_ratios:
        print(f"[Task {task_id}] Mass ratio = {ratio1:.2f}, Iyy ratio = {ratio2:.2f}")
        mass_nominal = 0.027 * m_nominal_ratio
        Iyy_nominal = 1.4e-5 * Iyy_nominal_ratio

        for target_x, target_z in targets:
            print(f"[Task {task_id}] Mass ratio = {ratio1:.2f}, Iyy ratio = {ratio2:.2f}, Target = ({target_x}, {target_z})")

            for episode_id in range(episodes_per_task):
                seed = task_id * 100 + episode_id
                env = Quadrotor(
                    gui=False,
                    ctrl_freq=50,
                    pyb_freq=50,
                    seed=seed,
                    done_on_out_of_bound=False,
                    quad_type=2,
                    inertial_prop=[mass_nominal * ratio1, Iyy_nominal * ratio2],
                    init_state_randomization_info={
                        'init_x': {'distrib': 'uniform', 'low': -1.0, 'high': 1.0},
                        'init_x_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
                        'init_z': {'distrib': 'uniform', 'low': 0.5, 'high': 1.5},
                        'init_z_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
                        'init_theta': {'distrib': 'uniform', 'low': -0.2, 'high': 0.2},
                        'init_theta_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1}
                    }
                )

                model = QuadrotorNominalDynamics(env).model()
                nominal_func = cs.Function("f_nom", [model.x, model.u], [model.f_expl])
                solver = MPC(model, N=N, t_horizon=t_horizon).solver

                obs, _ = env.reset()
                x = obs[:6]
                episode_data = []

                for step in range(Steps):
                    for k in range(N):
                        y_ref_k = np.array([target_x, 0, target_z, 0, 0, 0, 0, 0])
                        solver.set(k, "yref", y_ref_k)
                    y_ref_terminal = np.array([target_x, 0, target_z, 0, 0, 0])
                    solver.set(N, "yref", y_ref_terminal)

                    solver.set(0, "lbx", x)
                    solver.set(0, "ubx", x)
                    solver.solve()
                    u = solver.get(0, "u")

                    next_obs, _, _, _ = env.step(u)
                    x_next = next_obs[:6]

                    x_ddot = (x_next[1] - x[1]) / dt
                    z_ddot = (x_next[3] - x[3]) / dt
                    theta_ddot = (x_next[5] - x[5]) / dt

                    x_dot_nom = nominal_func(x, u).full().flatten()
                    res_x_ddot = x_ddot - x_dot_nom[1]
                    res_z_ddot = z_ddot - x_dot_nom[3]
                    res_theta_ddot = theta_ddot - x_dot_nom[5]

                    episode_data.append([
                        task_id, episode_id, step * dt, ratio1, ratio2, target_x, target_z, 
                        *x, *u,                                                            
                        x_ddot, z_ddot, theta_ddot,                                         
                        x_dot_nom[1], x_dot_nom[3], x_dot_nom[5],                        
                        res_x_ddot, res_z_ddot, res_theta_ddot,                           
                        env.MASS                                                           
                    ])                                                                       
                    x = x_next

                df_ep = pd.DataFrame(episode_data, columns=[
                    "task_id", "episode_id", "time", "ratio1", "ratio2", "target_x", "target_z",
                    "x", "x_dot", "z", "z_dot", "theta", "theta_dot",
                    "u1", "u2",
                    "x_ddot_true", "z_ddot_true", "theta_ddot_true",
                    "x_ddot_nom", "z_ddot_nom", "theta_ddot_nom",
                    "res_x_ddot", "res_z_ddot", "res_theta_ddot",
                    "mass"
                ])
                all_data.append(df_ep)

            task_id += 1

df_all = pd.concat(all_data, ignore_index=True)
save_path = os.path.join(output_dir, "quadrotor_meta_residual_mpc.csv")
df_all.to_csv(save_path, index=False)
print(f"\n✅ Saved full dataset with tasks to: {save_path}")
