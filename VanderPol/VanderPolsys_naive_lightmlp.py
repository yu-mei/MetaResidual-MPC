import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
import os
import matplotlib.pyplot as plt
import time

torch.manual_seed(49)
np.random.seed(49)

# ------------------------------------------------------------------------------
# Define MLP for residual correction
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# Parameters
mu_real = 0.2  # real system (used to generate ground truth)
mu_nom = 0.7   # nominal model used in prediction
dt = 0.02
Tsim = 10
Steps = int(Tsim / dt)
t = np.linspace(0, Tsim, Steps + 1)

# ------------------------------------------------------------------------------
# True Van der Pol dynamics
def vdp_oscillator(x1, x2, mu):
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return dx1, dx2

# ------------------------------------------------------------------------------
# Residual MLP and L4CasADi wrapper
residual_mlp = MLP(input_dim=2, output_dim=1, hidden_dim=64, num_layers=2)
residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3)
residual_criterion = nn.MSELoss()
for param in residual_mlp.parameters():
    param.requires_grad = False
l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_vdp", mutable=True)

# ------------------------------------------------------------------------------
# Learned dynamics: nominal (eqn-based) + residual (MLP)
def learned_dyn(x_row):
    x1, x2 = x_row[0, 0], x_row[0, 1]
    dx1 = x2
    dx2_nom = mu_nom * (1 - x1**2) * x2 - x1
    dx2_residual = l4c_residual(x_row)
    return cs.horzcat(dx1, dx2_nom + dx2_residual)

# ------------------------------------------------------------------------------
# Initialize ground truth trajectory
x1 = [0.5]
x2 = [0.5]
obs_buffer = []

# Output folder
segment_dir = "results/segments_naiveLightmlp"
os.makedirs(segment_dir, exist_ok=True)
segments = []
opt_times = []

# ------------------------------------------------------------------------------
# Simulation loop
for i in range(Steps):
    # --- Ground-truth RK4 integration (mu_real) ---
    x1_i, x2_i = x1[-1], x2[-1]
    k1_x1, k1_x2 = vdp_oscillator(x1_i, x2_i, mu_real)
    k2_x1, k2_x2 = vdp_oscillator(x1_i + 0.5 * dt * k1_x1, x2_i + 0.5 * dt * k1_x2, mu_real)
    k3_x1, k3_x2 = vdp_oscillator(x1_i + 0.5 * dt * k2_x1, x2_i + 0.5 * dt * k2_x2, mu_real)
    k4_x1, k4_x2 = vdp_oscillator(x1_i + dt * k3_x1, x2_i + dt * k3_x2, mu_real)

    next_x1 = x1_i + (dt / 6) * (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1)
    next_x2 = x2_i + (dt / 6) * (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2)

    x1.append(next_x1)
    x2.append(next_x2)

    # --- Residual data collection ---
    if i > 0:
        dx2_real = (x2[-1] - x2[-2]) / dt
        dx2_nom = mu_nom * (1 - x1[-2]**2) * x2[-2] - x1[-2]
        residual = dx2_real - dx2_nom
        obs_buffer.append((x1[-2], x2[-2], residual))

    # --- Online training every 50 steps ---
    if i > 0 and i % 50 == 0 and len(obs_buffer) >= 50:
        data = np.array(obs_buffer[-50:])
        X_batch = torch.tensor(data[:, :2], dtype=torch.float32)
        y_batch = torch.tensor(data[:, 2:], dtype=torch.float32)

        for param in residual_mlp.parameters():
            param.requires_grad = True
        for _ in range(50):  # few gradient steps
            residual_optimizer.zero_grad()
            pred = residual_mlp(X_batch)
            loss = residual_criterion(pred, y_batch)
            loss.backward()
            residual_optimizer.step()

        for param in residual_mlp.parameters():
            param.requires_grad = False
        l4c_residual.update(residual_mlp)

    # --- Predict next 1 second using learned dynamics ---
    if i % int(1 / dt) == 0:
        segment = []

        x0 = cs.DM([[x1[-1], x2[-1]]]) + cs.DM([np.random.normal(0, 0.025, size=2)])
        xt = x0
        segment.append(np.array(xt).squeeze())

        start_time = time.time()

        for _ in range(int(1 / dt)):
            k1 = learned_dyn(xt)
            k2 = learned_dyn(xt + 0.5 * dt * k1)
            k3 = learned_dyn(xt + 0.5 * dt * k2)
            k4 = learned_dyn(xt + dt * k3)

            xt = xt + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            segment.append(np.array(xt).squeeze())

        elapsed = time.time() - start_time
        print(f"Segment {i // 50:02d} rollout time: {1000*elapsed:.2f} ms")

        seg_array = np.stack(segment, axis=0)
        segments.append(seg_array)
        np.save(os.path.join(segment_dir, f"seg_{i // 50}.npy"), seg_array)
        opt_times.append(elapsed)

print(f'Mean prediction time: {1000*np.mean(opt_times[1:])/50:.1f}ms -- {1/np.mean(opt_times[1:])*50:.0f}Hz')

# ------------------------------------------------------------------------------
# Save trajectory
real_states = np.stack([t, x1, x2], axis=1)
np.save("results/Real_Online.npy", real_states)
print("Saved Real_Online.npy and predicted segments.")

# ------------------------------------------------------------------------------
# Plotting
real = np.load("results/Real_Online.npy")
t = real[:, 0]
x1 = real[:, 1]
x2 = real[:, 2]

segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith(".npy")])
segments = [np.load(os.path.join(segment_dir, f)) for f in segment_files]

plt.figure(figsize=(10, 4))

# Time series
plt.subplot(1, 2, 1)
plt.plot(t, x1, label=r'$x_1$ (Real)', linewidth=2, color='C0')
plt.plot(t, x2, label=r'$x_2$ (Real)', linewidth=2, color='C3')

for i, seg in enumerate(segments):
    seg_t = t[i * 50: i * 50 + 51]
    plt.plot(seg_t, seg[:, 0], '--', color='black', linewidth=1.2)
    plt.plot(seg_t, seg[:, 1], '--', color='black', linewidth=1.2)
    plt.plot(seg_t[0], seg[0, 0], 'o', color='C0')
    plt.plot(seg_t[0], seg[0, 1], 'o', color='C3')

plt.plot([], [], '--', color='black', label='Predicted')
plt.xlabel("Time [s]")
plt.ylabel("State")
plt.title("Time Series with Predicted Segments")
plt.legend()
plt.grid(True)

# Phase portrait
plt.subplot(1, 2, 2)
plt.plot(x1, x2, label='Real', linewidth=2, color='C0')
dx_real = x1[-1] - x1[-2]
dy_real = x2[-1] - x2[-2]
plt.arrow(x1[-2], x2[-2], dx_real, dy_real, head_width=0.2, head_length=0.2, fc='C0', ec='C0')

for seg in segments:
    plt.plot(seg[:, 0], seg[:, 1], '--', color='black')
    plt.plot(seg[0, 0], seg[0, 1], 'o', color='C0')
    dx = seg[-1, 0] - seg[-2, 0]
    dy = seg[-1, 1] - seg[-2, 1]
    plt.arrow(seg[-2, 0], seg[-2, 1], dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')

plt.plot([], [], '--', color='black', label='Predicted')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Phase Portrait with Predictions")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()



# ------------------------------------------------------------------------------
# RMSE Computing
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------------------
# Compute RMSE for Naive + Residual MLP
mlp_errors = []

for i, seg in enumerate(segments):
    seg_real = real[i * 50: i * 50 + 51, 1:3]  # shape (51, 2)
    rmse = np.sqrt(mean_squared_error(seg_real, seg))  # full segment RMSE
    mlp_errors.append(rmse)

mlp_errors = np.array(mlp_errors)

print(f"Naive + Residual MLP Avg RMSE: {mlp_errors.mean():.4f} ± {mlp_errors.std():.4f}")

# ------------------------------------------------------------------------------
# RMSE over segments plot
plt.figure(figsize=(8, 3))
plt.plot(mlp_errors, marker='o', label="Residual MLP RMSE", color='C1')
plt.xlabel("Segment Index")
plt.ylabel("RMSE")
plt.title("RMSE per Segment")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
