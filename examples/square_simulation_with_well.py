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
    animate_particle_positions_square,
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=12000,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=LineDefectHoppingCalculator(
            baserate=(0.01, 0.01 / 5), temperature=200
        ),
    )

    positions = run_simulation(params=params, rng_seed=1)

    isf_params = ISFParameters(params=params)
    amplitudes = get_amplitudes(isf_params=isf_params, positions=positions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_isf(
        x=amplitudes, t=params.times, ax=ax1, delta_k_index=36, isf_params=isf_params
    )

    dephasing_rates = get_dephasing_rates(amplitudes=amplitudes, t=params.times)

    plot_dephasing_rates(
        dephasing_rates=dephasing_rates,
        delta_k=isf_params.delta_k_array[:, 0],
        ax=ax2,
    )

    timesteps = np.arange(1, 12001)[::100]

    anim = animate_particle_positions_square(
        all_positions=positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    plt.show()
