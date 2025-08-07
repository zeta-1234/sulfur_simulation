"""Example simulation for a square lattice with a potential energy well across the center."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hopping_calculator import (
    SquareHoppingCalculatorWithWell,
)
from sulfur_simulation.isf import (
    ISFParameters,
    get_amplitude,
    get_dephasing_rates,
    plot_autocorrelation,
    plot_dephasing_rates,
)
from sulfur_simulation.scattering_calculation import (
    SimulationParameters,
    run_simulation,
)
from sulfur_simulation.show_simulation import (
    animate_particle_positions,
    get_timeframe_str,
)

if __name__ == "__main__":
    # Create a simulation with 3000 timesteps

    params = SimulationParameters(
        n_timesteps=3000,
        lattice_dimension=100,
        lattice_type="square",
        n_particles=500,
        rng_seed=2,
        temp=200,
    )

    hop_params = SquareHoppingCalculatorWithWell(baserate=0.01, params=params)

    isf_params = ISFParameters(
        n_delta_k_intervals=250,
        delta_k_max=2.5,
    )

    positions = run_simulation(params=params, hop_params=hop_params)

    print(
        get_timeframe_str(
            positions=positions, timestep=params.n_timesteps - 1, params=params
        )
    )

    amplitudes = get_amplitude(
        form_factor=isf_params.form_factor,
        delta_k=isf_params.delta_k_array,
        positions=positions,
        lattice_dimension=params.lattice_dimension,
    )

    average_amplitudes = np.mean(amplitudes, axis=1).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_autocorrelation(x=average_amplitudes[18], t=params.times, ax=ax1)

    dephasing_rates = get_dephasing_rates(amplitudes=average_amplitudes, t=params.times)

    plot_dephasing_rates(
        dephasing_rates=dephasing_rates,
        delta_k=isf_params.delta_k_array[:, 0],
        ax=ax2,
    )

    plt.show()

    timesteps = np.arange(1, 3001)[::10]

    anim = animate_particle_positions(
        all_positions=positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=params.lattice_spacing,
    )

    plt.show()
