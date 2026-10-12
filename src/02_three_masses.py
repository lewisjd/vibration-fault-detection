import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical properties
mass1 = 1.0  # kg
mass2 = 1.0  # kg
mass3 = 1.0  # kg
spring_stiffness1 = 100.0  # N/m
spring_stiffness2 = 100.0  # N/m
spring_stiffness3 = 100.0  # N/m
damping_coefficient = 0.2  # N·s/m

# Pull mass 3 by 0.01 m and release all masses from rest.
initial_displacement1 = 0.0  # m
initial_displacement2 = 0.0  # m
initial_displacement3 = 0.01  # m
initial_velocity1 = 0.0  # m/s
initial_velocity2 = 0.0  # m/s
initial_velocity3 = 0.0  # m/s
duration = 20.0  # s
measurement_interval = 0.01  # s

# Measurement times, including both endpoints.
number_of_points = int(round(duration / measurement_interval)) + 1
time_points = np.linspace(0, duration, number_of_points)


def motion_equation(t, state):
    x1, v1, x2, v2, x3, v3 = state

    # Each acceleration is the total spring and damping force / mass.
    dx1_dt = v1
    dv1_dt = (
        -damping_coefficient * v1
        - spring_stiffness1 * x1
        + spring_stiffness2 * (x2 - x1)
    ) / mass1

    dx2_dt = v2
    dv2_dt = (
        -damping_coefficient * v2
        - spring_stiffness2 * (x2 - x1)
        + spring_stiffness3 * (x3 - x2)
    ) / mass2

    dx3_dt = v3
    dv3_dt = (
        -damping_coefficient * v3
        - spring_stiffness3 * (x3 - x2)
    ) / mass3

    return [dx1_dt, dv1_dt, dx2_dt, dv2_dt, dx3_dt, dv3_dt]


# Solve the equations using displacement–velocity pairs.
solution = solve_ivp(
    motion_equation,
    [0, duration],
    [
        initial_displacement1, initial_velocity1,
        initial_displacement2, initial_velocity2,
        initial_displacement3, initial_velocity3,
    ],
    t_eval=time_points,
    rtol=1e-8,
    atol=1e-10,
)

if solution.success:
    plt.figure(figsize=(10, 5))

    # Displacements are stored in rows 0, 2 and 4.
    plt.plot(solution.t, solution.y[0], label="Mass 1 Displacement")
    plt.plot(solution.t, solution.y[2], label="Mass 2 Displacement")
    plt.plot(solution.t, solution.y[4], label="Mass 3 Displacement")
    plt.title("Three Mass-Spring System with Damping")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/02_three_mass_displacements.png", dpi=300)

    # Extract the quantities needed for mass 3's acceleration.
    x2 = solution.y[2]
    x3 = solution.y[4]
    v3 = solution.y[5]

    acceleration3 = (
        spring_stiffness3 * (x2 - x3)
        - damping_coefficient * v3
    ) / mass3

    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, acceleration3)
    plt.title("Healthy system — acceleration of mass 3")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/02_healthy_acceleration.png", dpi=300)
    plt.show()
else:
    print("Simulation failed:", solution.message)