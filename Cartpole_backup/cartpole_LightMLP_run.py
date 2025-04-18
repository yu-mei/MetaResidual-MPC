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
from safe_control_gym.envs.gym_control.cartpole import CartPole
torch.manual_seed(43)


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


class CartpoleLearnedDynamics:
    def __init__(self, gym_env, residual_model):
        self.gym_env = gym_env
        self.residual_model = residual_model

    def model(self):
        nominal_ratio = 0.75
        length = self.gym_env.EFFECTIVE_POLE_LENGTH * nominal_ratio
        m = self.gym_env.POLE_MASS * nominal_ratio 
        M = self.gym_env.CART_MASS * nominal_ratio 
        Mm, ml = m + M, m * length
        g = self.gym_env.GRAVITY_ACC
        dt = self.gym_env.CTRL_TIMESTEP

        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, x_dot, theta, theta_dot)
        U = cs.MX.sym('U')
        nx = 4
        nu = 1

        temp_factor = (U + ml * theta_dot**2 * cs.sin(theta)) / Mm
        theta_dot_dot = ((g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (length * (4.0 / 3.0 - m * cs.cos(theta)**2 / Mm)))
        X_dot_nominal = cs.vertcat(x_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot, theta_dot_dot)

        mlp_input = cs.vertcat(X, U)
        residual = self.residual_model(mlp_input.T).T
        X_dot_residual = cs.vertcat(0, residual[0], 0, residual[1])

        f_expl = X_dot_nominal + X_dot_residual
        x_start = np.array([0, 0, 0, 0])

        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 4)
        model.u = U
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal  
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "cartpole_learned"
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

        model_ac = self.acados_model(model=model)
        nx = 4
        nu = 1
        ny = nx + nu
        ny_e = nx

        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.Vx = np.eye(ny, nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.eye(nx)

        Q = 1*np.diag([5, 0.1, 5, 0.1])
        R = 1*np.diag([0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.constraints.x0 = model.x_start

        a_max = 10
        ocp.constraints.lbu = np.array([-a_max])
        ocp.constraints.ubu = np.array([a_max])
        ocp.constraints.idxbu = np.array([0])

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
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

    env_config = {
        'gui': False,
        'ctrl_freq': 50,
        'pyb_freq': 50,
        'seed': seed,
        'done_on_out_of_bound': True,
        'init_state_randomization_info': {
            'init_x': {'distrib': 'uniform', 'low': -2.0, 'high': 2.0},
            'init_x_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
            'init_theta': {'distrib': 'uniform', 'low': -0.2, 'high': 0.2},
            'init_theta_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1}
        }
    }

    env = CartPole(**env_config)
    obs, info = env.reset()
    xt = obs[:4]

    residual_mlp = MLP(input_dim=5, output_dim=2, hidden_dim=64, num_layers=3)
    for param in residual_mlp.parameters():
        param.requires_grad = False
    l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_pendulum", mutable=True)
    residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3)
    residual_criterion = nn.MSELoss()

    N = 20
    t_horizon = 1
    learned_model = CartpoleLearnedDynamics(env, l4c_residual)
    casadi_model = learned_model.model()
    nominal_func = cs.Function('nom', [casadi_model.x, casadi_model.u], [casadi_model.f_nominal])
    solver = MPC(model=learned_model.model(), N=N, t_horizon=t_horizon,
                    external_shared_lib_dir=l4c_residual.shared_lib_dir,
                    external_shared_lib_name=l4c_residual.name).solver

    dt = 1.0 / env_config['pyb_freq']
    Tsim = 10
    Steps = int(Tsim / dt)
    x_history, u_history, x_ref_history, theta_ref_history, opt_times = [xt], [], [], [], []
    obs_buffer = []
    batch_size = 20

    for i in range(Steps):
        current_time = i * dt
        for k in range(N):
            if k == 0:
                x_ref_history.append(0.0)
                theta_ref_history.append(0.0)
            solver.set(k, "yref", np.zeros(5))
        solver.set(N, "yref", np.zeros(4))
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        start = time.time()
        solver.solve()
        ut = solver.get(0, "u").item()
        u_history.append(ut)

        next_obs, _, _, _ = env.step(ut)
        xt = next_obs[:4]
        x_history.append(xt)

        if i > 0:
            dx2 = (x_history[-1][1] - x_history[-2][1]) / dt
            dx4 = (x_history[-1][3] - x_history[-2][3]) / dt
            obs_buffer.append((*xt, ut, dx2, dx4))

        if i > 0 and i % int(0.5 / dt) == 0 and len(obs_buffer) >= batch_size:
            data = np.array(obs_buffer[-batch_size:])
            X_batch = torch.tensor(data[:, :5], dtype=torch.float32)
            y_true = data[:, 5:]
            nominal = np.array([nominal_func(x[:4], [x[4]]).full().flatten() for x in data])
            y_nominal = nominal[:, [1, 3]]
            y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)

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

    min_length = min(len(u_history), len(x_history) - 1)
    df = pd.DataFrame({
        "time": np.linspace(0, Tsim, min_length),
        "x": np.array(x_history)[:min_length, 0],
        "x_dot": np.array(x_history)[:min_length, 1],
        "theta": np.array(x_history)[:min_length, 2],
        "theta_dot": np.array(x_history)[:min_length, 3],
        "u": np.array(u_history)[:min_length],
        "x_ref": np.array(x_ref_history)[:min_length],
        "theta_ref": np.array(theta_ref_history)[:min_length],
    })

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/lightmlp_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trajectory to {csv_path}")
