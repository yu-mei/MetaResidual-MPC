import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

np.random.seed(42)


# -----------------------------
# Data Collection Parameters
# -----------------------------
output_dir = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(output_dir, exist_ok=True)

mu_list = [round(0.1 * i, 1) for i in range(11)]
mu_nom = 0.2  # fixed nominal mu used for residual computation
dt = 0.02
T = 10
t_eval = np.arange(0, T, dt)
n_trajectories = 1
noise_std = 0.025

all_results = []

for mu in mu_list:
    print(f"Simulating for mu = {mu}")

    def vdp_oscillator(t, y):
        x1, x2 = y
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        return [dx1, dx2]

    for traj in range(n_trajectories):
        y0 = np.random.uniform(-2, 2, size=2)
        sol = solve_ivp(vdp_oscillator, [0, T], y0, t_eval=t_eval, method='RK45')

        noisy_x1 = sol.y[0] + np.random.normal(0, noise_std, size=sol.y[0].shape)
        noisy_x2 = sol.y[1] + np.random.normal(0, noise_std, size=sol.y[1].shape)

        x2_dot = np.gradient(noisy_x2, t_eval)

        # Nominal dynamics (no model/MLP)
        x2_dot_nom = mu_nom * (1 - noisy_x1**2) * noisy_x2 - noisy_x1

        # Residual = real - nominal
        residual = x2_dot - x2_dot_nom

        traj_df = pd.DataFrame({
            "trajectory": traj,
            "time": t_eval,
            "x1": noisy_x1,
            "x2": noisy_x2,
            "x2_dot": x2_dot,
            "x2_dot_nominal": x2_dot_nom,
            "residual": residual,
            "mu": mu
        })

        all_results.append(traj_df)

# Save everything
df_meta = pd.concat(all_results, ignore_index=True)
output_path = os.path.join(output_dir, "vdp_meta_nominal_residual.csv")
df_meta.to_csv(output_path, index=False)

print(f"Saved full meta-learning dataset with nominal residual to '{output_path}'")
