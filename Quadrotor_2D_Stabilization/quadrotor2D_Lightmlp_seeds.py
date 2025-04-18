import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor


COST = 'LINEAR_LS'  # NONLINEAR_LS
SAVE_FLAG = True
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)  


class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Quadrotor2DLearnedDynamics:
    def __init__(self, gym_env, residual_model):
        self.gym_env = gym_env
        self.residual_model = residual_model

    def model(self):
        # set up states & controls
        m_nominal_ratio = 0.66
        Iyy_nominal_ratio = 0.8
        m = self.gym_env.MASS * m_nominal_ratio         # self.gym_env.MASS = 0.027
        Iyy = self.gym_env.J[1, 1] * Iyy_nominal_ratio  # self.gym_env.J[1, 1] = 1.4e-5
        g, length = self.gym_env.GRAVITY_ACC, self.gym_env.L
        dt = self.gym_env.CTRL_TIMESTEP
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        # Input variables.
        T1 = cs.MX.sym('T1')
        T2 = cs.MX.sym('T2')
        U = cs.vertcat(T1, T2)
        nx, nu = 6, 2
        # Compute the input constraint
        n_mot = 4 / nu
        a_low = self.gym_env.KF * n_mot * (self.gym_env.PWM2RPM_SCALE * self.gym_env.MIN_PWM + self.gym_env.PWM2RPM_CONST)**2
        a_high = self.gym_env.KF * n_mot * (self.gym_env.PWM2RPM_SCALE * self.gym_env.MAX_PWM + self.gym_env.PWM2RPM_CONST)**2
        u_min= a_low * np.ones(nu)
        u_max = a_high * np.ones(nu)
        # Define dynamics equations.
        X_dot_nominal = cs.vertcat(x_dot,
                        cs.sin(theta) * (T1 + T2) / m, z_dot,
                        cs.cos(theta) * (T1 + T2) / m - g, theta_dot,
                        length * (T2 - T1) / Iyy / np.sqrt(2))
        
        mlp_input = cs.vertcat(X, U)
        residual = self.residual_model(mlp_input.T).T  # 8->3 (X, U)-> (x_ddot, z_ddot, theta_ddot)
        X_dot_residual = cs.vertcat(0, residual[0], 0, residual[1], 0, residual[2])

        f_expl = X_dot_nominal + X_dot_residual
        x_start = np.array([0, 0, 0.75, 0, 0, 0])


        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 6)
        model.u = U
        model.u_min = u_min
        model.u_max = u_max
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "quadrotor2D_learned"

        return model


class MPC:
    def __init__(self, model, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        self.model = model
        self.N = N
        self.t_horizon = t_horizon
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

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

        if COST == 'LINEAR_LS':
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
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'

            x = ocp.model.x
            u = ocp.model.u
            y_expr = cs.vertcat(x, u)

            # Trivial PyTorch index 0
            l4c_y_expr = l4c.L4CasADi(lambda y_expr: y_expr, name='y_expr')

            ocp.model.cost_y_expr = l4c_y_expr(y_expr)
            ocp.model.cost_y_expr_e = x

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
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name

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

# ----------------------------- Simulation Loop -----------------------------
for seed in range(31, 51):
    print(f"\n====== Running simulation with seed {seed} ======")
    # ------------------------------------------------------------------------------
    # Parameters
    # Configure environment parameters
    env_config = {
        'gui': False,  # Set to False for faster data collection
        'ctrl_freq': 50,  # Control frequency
        'pyb_freq': 50,  # Physics simulation frequency
        'seed': seed,
        'done_on_out_of_bound': True,  # Set to False if you want longer episodes
        
        'init_state_randomization_info': {
            'init_x': {'distrib': 'uniform', 'low': -1, 'high': 1},
            'init_x_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
            'init_z': {'distrib': 'uniform', 'low': 0.5, 'high': 1.5},
            'init_z_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
            'init_theta': {'distrib': 'uniform', 'low': -0.2, 'high': 0.2},
            'init_theta_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1}
        }

    }

    env = Quadrotor(**env_config)
    model = env.symbolic

    obs, info = env.reset()
    xt = obs[:6]  # [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]

    # ------------------------------------------------------------------------------
    # Residual MLP: lightweight
    residual_mlp = MLP(input_dim=6 + 2, output_dim=3, hidden_dim=64, num_layers=3)
    for param in residual_mlp.parameters():
        param.requires_grad = False
    l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_quadrotor2D", mutable=True)
    residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3)
    residual_criterion = nn.MSELoss()

    # ------------------------------------------------------------------------------
    # MPC Setup
    N = 20
    t_horizon = 1
    learned_model = Quadrotor2DLearnedDynamics(env, l4c_residual)
    casadi_model = learned_model.model()
    nominal_func = cs.Function('nom', [casadi_model.x, casadi_model.u], [casadi_model.f_nominal])
    solver = MPC(model=learned_model.model(), N=N, t_horizon=t_horizon,
                    external_shared_lib_dir=l4c_residual.shared_lib_dir,
                    external_shared_lib_name=l4c_residual.name).solver

    # ------------------------------------------------------------------------------
    # Simulation Setup
    dt = 1.0 / env_config['pyb_freq']  # Time step
    Tsim = 10
    Steps = int(Tsim / dt)
    x_history, u_history, x_ref_history, opt_times = [xt], [], [], []

    # Residual Finetune
    obs_buffer = []  # stores (x, x_dot, z, z_dot, theta, theta_dot, u1, u2)
    batch_size = 32

    for i in range(Steps):
        current_time = i * dt

        # Set reference for each step in MPC horizon (all zeros)
        for k in range(N):
            if k == 0:  # Only store the first reference of each MPC horizon
                x_ref_history.append([0, 1, 0])
                
            # Set reference for state [x, x_dot, z, z_dot, theta, theta_dot, u1, u2]
            y_ref_k = np.array([0, 0, 1, 0, 0, 0, 0, 0])  # All zeros for reference
            solver.set(k, "yref", y_ref_k)
        # Set terminal reference (only state)
        y_ref_terminal = np.array([0, 0, 1, 0, 0, 0])  # All zeros for terminal reference
        solver.set(N, "yref", y_ref_terminal)

        start = time.time()
        # Apply current state as constraint
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        # Solve MPC and apply control
        solver.solve()
        ut = solver.get(0, "u")
        u_history.append(ut)
        
        # Since simulation_step seems to be missing, let's use the environment step
        next_obs, reward, done, info = env.step(ut)
        xt = next_obs[:6]
        
        # Add measurement noise (now with correct size=4)
        xt_measured = xt + np.random.normal(0, 0.01, size=6)
        #xt = xt_measured
        x_history.append(xt)

        # --------------------Residual----------------------#
        # Compute x_ddot theta_ddot from difference
        if i > 0:
            dx2 = (x_history[-1][1] - x_history[-2][1]) / dt
            dz2 = (x_history[-1][3] - x_history[-2][3]) / dt
            dtheta2 = (x_history[-1][5] - x_history[-2][5]) / dt  # theta_ddot

            # Append: [state (6), action (2), measured accel (3)]
            obs_buffer.append((*xt, *ut, dx2, dz2, dtheta2))

        if i > 0 and i % int(0.5 / dt) == 0 and len(obs_buffer) >= batch_size:
            data = np.array(obs_buffer[-batch_size:])

            # Split into input features and target
            X_batch = torch.tensor(data[:, :8], dtype=torch.float32) # [x, x_dot, z, z_dot, theta, theta_dot, u1, u2]
            y_true = data[:, 8:]                                     # [x_ddot_true, z_ddot_true, theta_ddot_true]

            # Evaluate nominal accelerations (use symbolic nominal model)
            # Nominal inputs = x[0:6], u[6:8]
            nominal = np.array([
                nominal_func(x[:6], x[6:8]).full().flatten() for x in data
            ])
            y_nominal = nominal[:, [1, 3, 5]]  # x_ddot, z_ddot, theta_ddot

            # Compute target residual: measured - nominal
            y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)

            # Train residual MLP
            for p in residual_mlp.parameters(): p.requires_grad = True
            for _ in range(20):
                residual_optimizer.zero_grad()
                pred = residual_mlp(X_batch)
                loss = residual_criterion(pred, y_target)
                loss.backward()
                residual_optimizer.step()
            for p in residual_mlp.parameters(): p.requires_grad = False
            l4c_residual.update(residual_mlp)

        elapsed = time.time() - start
        opt_times.append(elapsed)

    # Convert to numpy arrays for easier indexing
    x_history = np.array(x_history)
    u_history = np.array(u_history)
    x_ref_history = np.array(x_ref_history)

    # Create time grids with matching dimensions
    t_grid_states = np.linspace(0, Tsim, len(x_history))
    t_grid_inputs = np.linspace(0, Tsim, len(u_history))

    print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')
    print(f'State history shape: {x_history.shape}, Control history shape: {u_history.shape}')

    # ------------------------------------------------------------------------------
    # Plot
    plt.figure(figsize=(15, 10))

    # Plot x position
    plt.subplot(2, 1, 1)
    plt.plot(t_grid_states, x_history[:, 0], linewidth=2, color='C1', label='x')
    plt.plot(t_grid_inputs, x_ref_history[:, 0], '--', linewidth=2, label='x_ref', color='C0', alpha=0.7)
    plt.ylabel('x [m]')
    plt.legend()
    plt.grid()

    # Plot z position
    plt.subplot(2, 1, 2)
    plt.plot(t_grid_states, x_history[:, 2], linewidth=2, color='C7', label='z')
    plt.plot(t_grid_inputs, x_ref_history[:, 1], '--', linewidth=2, label='z_ref', color='C0', alpha=0.7)
    plt.ylabel('z [m]')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    if SAVE_FLAG:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{timestamp}.png"
        # plt.savefig(plot_path, dpi=300)
        print(f"Saved plot to {plot_path}")

    #plt.show()


    # ------------------------------------------------------------------------------
    # Save Results
    if SAVE_FLAG:
        min_length = min(len(t_grid_inputs), len(x_history)-1)

        x = x_history[:min_length, 0]
        x_dot = x_history[:min_length, 1]
        z = x_history[:min_length, 2]
        z_dot = x_history[:min_length, 3]
        theta = x_history[:min_length, 4]
        theta_dot = x_history[:min_length, 5]
        u_data = u_history[:min_length]
        x_ref_data = x_ref_history[:min_length, 0]
        z_ref_data = x_ref_history[:min_length, 1]
        theta_ref_data = x_ref_history[:min_length, 2]
        time_data = t_grid_inputs[:min_length]

        df = pd.DataFrame({
            "time": time_data,
            "x": x,
            "x_dot": x_dot,
            "z": z,
            "z_dot": z_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "u1": u_data[:, 0],
            "u2": u_data[:, 1],
            "x_ref": x_ref_data,
            "z_ref": z_ref_data,
            "theta_ref": theta_ref_data
        })

        seed = env_config['seed']
        csv_path = f"results/lightmlp_seed{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved trajectory to {csv_path}")
