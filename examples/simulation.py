from __future__ import annotations

import numpy as np

from sulfur_simulation.scattering_calculation import (
    SimulationParameters,
    update_position,
)

if __name__ == "__main__":
    # Create a simulation with 5000 timesteps
    params = SimulationParameters(
        total_scattering_angle=1.2,
        angle_of_incidence=0.8,
        n_timesteps=5000,
        step=1,
        incident_wavevector=np.array([3, 6]),
    )

    rng = np.random.default_rng()  # create generator for random numbers
    position = np.array([0, 0])  # defines the initial position
    n_steps = params.n_timesteps // params.step + 1  # calculates the number of steps

    positions = np.empty((n_steps, 2))  # generates empty array for positions
    positions[0] = position

    for index in range(
        1, n_steps
    ):  # loops over update_position() and saves result to positions array
        position = update_position(
            position=position,
            hopping_probability=params.hopping_probability,
            lattice_spacing=params.lattice_spacing,
            rng=rng,
        )
        positions[index] = position

    time = np.arange(
        0, params.n_timesteps + 1, params.step
    )  # lists times for steps for later plotting

    print(positions[:, 0])  # x positions
    print(positions[:, 1])  # y positions
    print(time)
