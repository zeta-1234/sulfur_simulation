"""Example simulation for a square lattice with a Lennard Jones interacting potential."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hexagonal_hopping_calculator import (
    HexagonalInteractingHoppingCalculator,
    get_lennard_jones_potential,
)
from sulfur_simulation.isf import (
    ISFParameters,
    get_dephasing_rates,
    get_multiple_amplitudes,
    plot_dephasing_rates,
    plot_isf,
)
from sulfur_simulation.scattering_calculation import (
    SimulationParameters,
    run_multiple_simulations,
)
from sulfur_simulation.show_simulation import (
    animate_particle_positions_skewed,
    plot_mean_jump_rates,
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=1000,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=HexagonalInteractingHoppingCalculator(
            baserate=0.01,
            temperature=200,
            lattice_spacing=2.5,
            interaction=get_lennard_jones_potential(sigma=2.55, epsilon=0.03 * 1.6e-19),
        ),
    )

    results = run_multiple_simulations(n_runs=5, params=params)

    isf_params = ISFParameters(params=params)

    all_amplitudes = get_multiple_amplitudes(isf_params=isf_params, results=results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_isf(
        x=all_amplitudes,
        t=params.times,
        ax=axes[0],
        delta_k_index=50,
        isf_params=isf_params,
    )

    dephasing_rates = get_dephasing_rates(amplitudes=all_amplitudes, t=params.times)

    plot_dephasing_rates(
        dephasing_rates=dephasing_rates,
        delta_k=isf_params.delta_k_array[:, 0],
        ax=axes[1],
    )

    plot_mean_jump_rates(results=results, ax=axes[2])

    timesteps = np.arange(1, 100, 1, dtype=int)

    anim = animate_particle_positions_skewed(
        all_positions=results[0].positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        dx_dy=(2.5, 2.5 * np.sqrt(3) / 2),
    )

    plt.show()
