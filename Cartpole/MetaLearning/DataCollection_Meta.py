import os
import numpy as np
import pandas as pd
import casadi as cs
import matplotlib.pyplot as plt
from safe_control_gym.envs.gym_control.cartpole import CartPole
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import torch
import torch.nn as nn
import l4casadi as l4c
import scipy.linalg
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ---------------------------------------------
# Dummy Residual MLP for Acados
# ---------------------------------------------
class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 2), nn.Tanh())

    def forward(self, x):
        return torch.zeros((x.shape[0], 2))

# ---------------------------------------------
# CartPole Nominal Dynamics (fixed 0.75 scaling)
# ---------------------------------------------
class CartpoleNominalDynamics:
    def __init__(self, gym_env):
        self.gym_env = gym_env

    def model(self):

        inertia_nom = {'pole_length': 0.5*0.66, 'cart_mass': 1.0*0.66, 'pole_mass': 0.1*0.66}
        length = inertia_nom['pole_length']
        m = inertia_nom['pole_mass']
        M = inertia_nom['cart_mass']

        Mm, ml = m + M, m * length
        g = self.gym_env.GRAVITY_ACC

        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        u = cs.MX.sym('u')

        X = cs.vertcat(x, x_dot, theta, theta_dot)
        U = u

        temp = (u + ml * theta_dot ** 2 * cs.sin(theta)) / Mm
        theta_ddot = (g * cs.sin(theta) - cs.cos(theta) * temp) / (
            length * (4.0 / 3.0 - m * cs.cos(theta) ** 2 / Mm)
        )
        x_ddot = temp - ml * theta_ddot * cs.cos(theta) / Mm

        f_expl = cs.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 4)
        model.u = U
        model.f_expl = f_expl
        model.name = "cartpole_nominal"
        model.x_start = np.zeros(4)
        model.p = cs.vertcat([])
        model.z = cs.vertcat([])
        model.constraints = cs.vertcat([])

        return model

# ---------------------------------------------
# MPC Setup
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
        model_ac = self.acados_model(model)

        ocp = AcadosOcp()
        ocp.model = model_ac

        nx, nu = 4, 1
        ny = nx + nu
        ny_e = nx

        ocp.dims.N = self.N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = self.t_horizon

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.Vx = np.zeros((ny, nx))
        np.fill_diagonal(ocp.cost.Vx, 1)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.eye(nx)

        Q = 1 * np.diag([5, 0.1, 5, 0.1])
        R = 1 * np.diag([0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.x0 = model.x_start
        ocp.constraints.lbu = np.array([-10])
        ocp.constraints.ubu = np.array([10])
        ocp.constraints.idxbu = np.array([0])

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
# Simulation Loop
# ---------------------------------------------
output_dir = "meta_dataset_mpc"
os.makedirs(output_dir, exist_ok=True)

dt = 0.02
T = 10.0
Steps = int(T / dt)
N = 20
t_horizon = 1.0
theta_limit = np.pi / 2

episodes_per_task = 5
#ratios = np.round(np.linspace(1.0, 2.0, 10), 3)
ratios = np.round(np.linspace(0.75, 2.0, 10), 3)
all_data = []

for task_id, ratio in enumerate(ratios):
    print(f"[Task {task_id}] Ratio = {ratio:.2f}")
    inertia_nom = {'pole_length': 0.5*0.66, 'cart_mass': 1.0*0.66, 'pole_mass': 0.1*0.66}
    inertia = {
        'pole_length': inertia_nom['pole_length'] * ratio,
        'cart_mass': inertia_nom['cart_mass'] * ratio,
        'pole_mass': inertia_nom['pole_mass'] * ratio
    }

    for episode_id in range(episodes_per_task):
        seed = task_id * 100 + episode_id

        env = CartPole(
            gui=False,
            ctrl_freq=50,
            pyb_freq=50,
            seed=seed,
            done_on_out_of_bound=False,
            episode_len_sec= 20,  #Set to longer than the simulation time length, so that the returned 'done' is only based on the success of the stabalization
            inertial_prop=inertia,
            init_state_randomization_info={
                'init_x': {'distrib': 'uniform', 'low': -2.0, 'high': 2.0},
                'init_x_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
                'init_theta': {'distrib': 'uniform', 'low': -0.2, 'high': 0.2},
                'init_theta_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1}
            }
        )

        mpc_model = CartpoleNominalDynamics(env).model()
        nominal_func = cs.Function('f_nominal', [mpc_model.x, mpc_model.u], [mpc_model.f_expl])
        solver = MPC(model=mpc_model, N=N, t_horizon=t_horizon).solver

        obs, _ = env.reset()
        x = obs[:4]
        episode_data = []

        for step in range(Steps):
            for k in range(N):
                solver.set(k, "yref", np.zeros(5))
            solver.set(N, "yref", np.zeros(4))
            solver.set(0, "lbx", x)
            solver.set(0, "ubx", x)
            solver.solve()
            u = solver.get(0, "u").item()

            x_next_obs, _, _, _ = env.step(u)
            x_next = x_next_obs[:4]

            x_ddot = (x_next[1] - x[1]) / dt
            theta_ddot = (x_next[3] - x[3]) / dt

            x_dot_nom = nominal_func(x, [u]).full().flatten()
            res_x_ddot = x_ddot - x_dot_nom[1]
            res_theta_ddot = theta_ddot - x_dot_nom[3]

            episode_data.append([
                task_id, episode_id, step * dt, *x, u, x_ddot, theta_ddot,
                x_dot_nom[1], x_dot_nom[3], res_x_ddot, res_theta_ddot,
                inertia['pole_length'], inertia['cart_mass'], inertia['pole_mass']
            ])

            if abs(x_next[2]) > theta_limit:
                print(f"[WARN] Terminated early at step {step} due to theta overflow.")
                break

            x = x_next

        df_ep = pd.DataFrame(episode_data, columns=[
            "task_id", "episode_id", "time", "x", "x_dot", "theta", "theta_dot", "u",
            "x_ddot_true", "theta_ddot_true", "x_ddot_nom", "theta_ddot_nom",
            "res_x_ddot", "res_theta_ddot",
            "pole_length", "cart_mass", "pole_mass"
        ])
        all_data.append(df_ep)

df_all = pd.concat(all_data, ignore_index=True)
save_path = os.path.join(output_dir, "cartpole_meta_residual_mpc.csv")
df_all.to_csv(save_path, index=False)
print(f"\n✅ Saved full dataset with tasks to: {save_path}")