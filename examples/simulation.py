from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.isf import (
    get_dephasing_rates,
    plot_autocorrelation,
    plot_dephasing_rates,
)
from sulfur_simulation.scattering_calculation import (
    ResultParameters,
    SimulationParameters,
    get_10_delta_k,
    get_amplitude,
    update_position,
)

if __name__ == "__main__":
    # Create a simulation with 5000 timesteps
    params = SimulationParameters(
        n_timesteps=2000,
        step=1,
    )

    result_params = ResultParameters(n_delta_k_intervals=500, delta_k_max=2.5)

    if result_params.delta_k_min == 0:
        msg = "delta_k_min should not be zero"
        raise ValueError(msg)

    delta_k_array = get_10_delta_k(
        n_delta_k=result_params.n_delta_k_intervals,
        min_delta_k=result_params.delta_k_min,
        max_delta_k=result_params.delta_k_max,
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

    amplitudes = get_amplitude(  # calculating scattered amplitude from positions
        delta_k=delta_k_array,
        position=positions,
        form_factor=params.form_factor,
    )

    fig, ax = plot_autocorrelation(x=amplitudes[100], t=time)

    plt.show()

    dephasing_rates = get_dephasing_rates(amplitudes=amplitudes, t=time)

    fig, ax = plot_dephasing_rates(
        dephasing_rates=dephasing_rates, delta_k=delta_k_array
    )

    plt.show()
