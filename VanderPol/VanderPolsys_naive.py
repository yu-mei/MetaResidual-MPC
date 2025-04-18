import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
import os
import matplotlib.pyplot as plt
import random
import time

torch.manual_seed(49)
np.random.seed(49)


# ------------------------------------------------------------------------------
# Parameters
mu_real = 0.2
mu_nominal = 0.7
dt = 0.02
Tsim = 10
Steps = int(Tsim / dt)
t = np.linspace(0, Tsim, Steps + 1)


# ------------------------------------------------------------------------------
# True Van der Pol dynamics (for RK4)
def vdp_oscillator(x1, x2, mu):
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return dx1, dx2

# ------------------------------------------------------------------------------
# Initialize ground truth
x1 = [0.5]
x2 = [0.5]

# Output folder
segment_dir = "results/segments_naive"
os.makedirs(segment_dir, exist_ok=True)
segments = []
opt_times = []

# ------------------------------------------------------------------------------
# Simulation loop
for i in range(Steps):
    # --- True RK4 integration step ---
    x1_i, x2_i = x1[-1], x2[-1]
    k1_x1, k1_x2 = vdp_oscillator(x1_i, x2_i, mu_real)
    k2_x1, k2_x2 = vdp_oscillator(x1_i + 0.5 * dt * k1_x1, x2_i + 0.5 * dt * k1_x2, mu_real)
    k3_x1, k3_x2 = vdp_oscillator(x1_i + 0.5 * dt * k2_x1, x2_i + 0.5 * dt * k2_x2, mu_real)
    k4_x1, k4_x2 = vdp_oscillator(x1_i + dt * k3_x1, x2_i + dt * k3_x2, mu_real)

    next_x1 = x1_i + (dt / 6) * (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1)
    next_x2 = x2_i + (dt / 6) * (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2)

    x1.append(next_x1)
    x2.append(next_x2)

    # --- Every 1 second: predict next 1s using MLP dynamics ---
    if i % int(1 / dt) == 0:
        segment = []

        # Create noisy initial condition (row vector shape)
        # cs.DM([1, 2]) default column vector
        # cs.DM([[1, 2]]) default vector vector
        x0 = cs.DM([[x1[-1], x2[-1]]]) + cs.DM([np.random.normal(0, 0.025, size=2)]) #imagine the N(0, 0.025^2) noise 
        xt = x0
        segment.append(np.array(xt).squeeze())  # store initial

        start_time = time.time()

        # RK4 rollout on nominal dynamics
        for _ in range(int(1 / dt)):
            x1_p, x2_p = xt[0, 0], xt[0, 1]

            k1_x1, k1_x2 = vdp_oscillator(x1_p, x2_p, mu_nominal)
            k2_x1, k2_x2 = vdp_oscillator(x1_p + 0.5 * dt * k1_x1, x2_p + 0.5 * dt * k1_x2, mu_nominal)
            k3_x1, k3_x2 = vdp_oscillator(x1_p + 0.5 * dt * k2_x1, x2_p + 0.5 * dt * k2_x2, mu_nominal)
            k4_x1, k4_x2 = vdp_oscillator(x1_p + dt * k3_x1, x2_p + dt * k3_x2, mu_nominal)

            next_x1_p = x1_p + (dt / 6) * (k1_x1 + 2 * k2_x1 + 2 * k3_x1 + k4_x1)
            next_x2_p = x2_p + (dt / 6) * (k1_x2 + 2 * k2_x2 + 2 * k3_x2 + k4_x2)

            xt = cs.DM([[float(next_x1_p), float(next_x2_p)]])
            segment.append(np.array(xt).squeeze())

        seg_array = np.stack(segment, axis=0)  # shape (51, 2)
        segments.append(seg_array)
        np.save(os.path.join(segment_dir, f"seg_{i // 50}.npy"), seg_array)

        # End timing here
        elapsed = time.time() - start_time
        print(f"Segment {i // 50:02d} rollout time: {1000*elapsed:.4f} ms")
        opt_times.append(elapsed)

print(f'Mean prediction time: {1000*np.mean(opt_times[1:])/50:.1f}ms -- {1/np.mean(opt_times[1:])*50:.0f}Hz)')

# Save true state trajectory
real_states = np.stack([t, x1, x2], axis=1)  # shape: (N+1, 3)
np.save("results/Real_Online.npy", real_states)
print("Saved Real_Online.npy and all predicted segments.")

# ------------------------------------------------------------------------------
# Plotting
segment_dir = f"results/segments_naive"
real_path = "results/Real_Online.npy"

real = np.load(real_path)
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
    plt.plot(seg_t, seg[:, 0], '--', color='black', linewidth=1.5)
    plt.plot(seg_t, seg[:, 1], '--', color='black', linewidth=1.5)
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
plt.show()
