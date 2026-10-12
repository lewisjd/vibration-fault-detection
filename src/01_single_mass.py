import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical variables
mass = 1.0  # kg
spring_stiffness = 100.0  # N/m
damping_coefficient = 0.2  # N·s/m
initial_displacement = 0.01  # m
initial_velocity = 0.0  # m/s
duration = 20.0  # s
measurement_interval = 0.01  # s

# Include both endpoints: 2,001 points give 2,000 intervals.
number_of_points = int(round(duration / measurement_interval)) + 1
time_points = np.linspace(0, duration, number_of_points)


def motion_equation(t, state):
    x, v = state

    dx_dt = v
    dv_dt = (-damping_coefficient * v - spring_stiffness * x) / mass

    return [dx_dt, dv_dt]


solution = solve_ivp(
    motion_equation,
    [0, duration],
    [initial_displacement, initial_velocity],
    t_eval=time_points,
    rtol=1e-8,
    atol=1e-10,
)

if solution.success:
    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, solution.y[0], label="Displacement")
    plt.title("Single Mass-Spring System with Damping")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/01_single_mass.png", dpi=300)
    plt.show()
else:
    print("Simulation failed:", solution.message)