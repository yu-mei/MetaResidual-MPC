import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
SAVE_FLAG = True

# Parameters
mu = 0.2
dt = 0.02  # time step
T = 10     # total simulation time
t_eval = np.arange(0, T, dt)

# Dynamics function
def vdp_oscillator(t, y):
    x1, x2 = y
    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1
    return [dx1, dx2]

# Initial condition
y0 = [0.5, 0.5]

# Solve using advanced ODE solver (RK45 is default)
Real = solve_ivp(vdp_oscillator, [0, T], y0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(Real.t, Real.y[0], label='$x_1(t)$')
plt.plot(Real.t, Real.y[1], label='$x_2(t)$')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Van der Pol Oscillator Time Series')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Real.y[0], Real.y[1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Phase Portrait')
plt.grid(True)

plt.tight_layout()
plt.show()

if SAVE_FLAG:
    import os
    os.makedirs("results", exist_ok=True)
    np.savez(f"results/Real_vdp_mu{mu}.npz", t=Real.t, x1=Real.y[0], x2=Real.y[1])