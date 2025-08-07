"""Example simulation for a square lattice with a potential energy well across the center."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hopping_calculator import (
    LineDefectHoppingCalculator,
)
from sulfur_simulation.isf import (
    ISFParameters,
    get_amplitudes,
    get_dephasing_rates,
    plot_dephasing_rates,
    plot_isf,
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
    params = SimulationParameters(
        n_timesteps=3000,
        lattice_dimension=(100, 100),
        n_particles=500,
        rng_seed=2,
        hopping_calculator=LineDefectHoppingCalculator(baserate=0.01, temperature=200),
    )

    isf_params = ISFParameters(
        n_delta_k_intervals=250,
        delta_k_max=2.5,
    )

    positions = run_simulation(params=params)

    print(
        get_timeframe_str(
            positions=positions, timestep=params.n_timesteps - 1, params=params
        )
    )

    amplitudes = get_amplitudes(params=isf_params, positions=positions)

    average_amplitudes = np.mean(amplitudes, axis=1).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_isf(x=average_amplitudes[18], t=params.times, ax=ax1)

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
