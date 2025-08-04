from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.isf import (
    ISFParameters,
    get_amplitude,
    get_dephasing_rates,
    plot_autocorrelation,
    plot_dephasing_rates,
)
from sulfur_simulation.scattering_calculation import (
    SimulationParameters,
    run_multiple_simulations,
)

if __name__ == "__main__":
    # Create a simulation with 5000 timesteps

    params = SimulationParameters(n_timesteps=2000, initial_position=np.array([0, 0]))

    isf_params = ISFParameters(n_delta_k_intervals=250, delta_k_max=2.5)

    positions = run_multiple_simulations(params, n_runs=700)

    amplitudes = get_amplitude(
        form_factor=isf_params.form_factor,
        delta_k=isf_params.delta_k_array,
        position=positions,
    )

    average_amplitudes = np.mean(amplitudes, axis=0).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_autocorrelation(x=average_amplitudes[85], t=params.times, ax=ax1)

    dephasing_rates = get_dephasing_rates(amplitudes=average_amplitudes, t=params.times)

    plot_dephasing_rates(
        dephasing_rates=dephasing_rates,
        delta_k=isf_params.delta_k_array[:, 0],
        ax=ax2,
    )

    plt.show()
