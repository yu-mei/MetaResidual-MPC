import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


class OscillatorLearnedDyanmics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        x    = cs.MX.sym('x', 2)       #state[x1, x2]
        x_dot = cs.MX.sym('xdot', 2)

        res_model = self.learned_dyn(x)
        p = self.learned_dyn.get_sym_params()
        parameter_values = self.learned_dyn.get_params(np.array([0, 0]))

        f_expl = cs.vertcat(
                x[1],        # x1_dot
                res_model[0] # learned x1_dot_dot
                )

        x_start = np.array([0.3, 0.5])

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = x
        model.xdot = x_dot
        model.u = cs.vertcat([])
        model.z = cs.vertcat([])
        model.p = p
        model.parameter_values = parameter_values
        model.f_expl = f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "oscillator_learned"

        return model


class MPC:
    def __init__(self, model, Tf):
        self.model = model
        self.Tf = Tf

    @property
    def simulator(self):
        return AcadosSimSolver(self.sim())

    def sim(self):
        model = self.model
        Tf = self.Tf

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.nx = nx
        
        # set simulation time
        sim.solver_options.T = Tf
        # set options
        sim.solver_options.integrator_type = 'IRK'
        sim.solver_options.num_stages = 3
        sim.solver_options.num_steps = 3
        sim.solver_options.newton_iter = 3 # for implicit integrator
        sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

        sim.parameter_values = model.parameter_values

        return sim

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac


def run(order = 1):
    Tf = 0.02

    # Create the learned dynamics model (the MLP for pendulum dynamics)
    checkpoint = torch.load("MLP/mlp_vdp_checkpoint_5_256.pth")
    loaded_model = MLP(
        input_dim=checkpoint['input_dim'],
        output_dim=checkpoint['output_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    learned_dyn_model = l4c.realtime.RealTimeL4CasADi(loaded_model, approximation_order=order)

    model = OscillatorLearnedDyanmics(learned_dyn_model)
    simulator = MPC(model=model.model(), Tf=Tf).simulator

    print('Warming up model...')
    x_l = []
    x_l.append(simulator.get("x"))
    for i in range(20):
        learned_dyn_model.get_params(np.stack(x_l, axis=0))
    print('Warmed up!')

    SimX = []
    SimTime = 10
    xt = np.array([0.3, 0.5])
    opt_times = []

    for i in range(int(SimTime/Tf)):
        now = time.time()

        current_state = xt
        simulator.set("x", current_state)
        simulator.set("p", learned_dyn_model.get_params(xt))  # if applicable
        simulator.solve()
        xt = simulator.get("x")
        SimX.append(xt)

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time for Order {order}: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

    # Convert SimX list to a NumPy array
    SimX = np.array(SimX)  # shape: [N_steps, 2]
    time_vec = np.linspace(0, SimTime, len(SimX))

    # Plotting
    plt.figure(figsize=(10, 4))

    # Time Series
    plt.subplot(1, 2, 1)
    plt.plot(time_vec, SimX[:, 0], label='$x_1(t)$')
    plt.plot(time_vec, SimX[:, 1], label='$x_2(t)$')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title('Van der Pol Oscillator Time Series')
    plt.legend()

    # Phase Portrait
    plt.subplot(1, 2, 2)
    plt.plot(SimX[:, 0], SimX[:, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Phase Portrait')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run(order=2)