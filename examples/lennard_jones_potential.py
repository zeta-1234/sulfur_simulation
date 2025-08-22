"""Example simulation for a square lattice with a Lennard Jones interacting potential."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hopping_calculator import (
    InteractingHoppingCalculator,
    SquareBaseRate,
    get_lennard_jones_potential,
)
from sulfur_simulation.isf import (
    ISFParameters,
    get_all_amplitudes,
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
    plot_mean_jump_rates,
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=12000,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=InteractingHoppingCalculator(
            baserate=SquareBaseRate(straight_rate=0.01, diagonal_rate=0.002),
            temperature=200,
            lattice_properties=(2.5, np.array([1, 0]), np.array([0, 1])),
            interaction=get_lennard_jones_potential(sigma=2.55, epsilon=0.03 * 1.6e-19),
        ),
    )

    results = run_simulation(n_runs=5, params=params)

    isf_params = ISFParameters(params=params)

    all_amplitudes = get_all_amplitudes(isf_params=isf_params, results=results)

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

    timesteps = np.arange(1, params.n_timesteps, 20, dtype=int)

    anim = animate_particle_positions_square(
        all_positions=results[0].positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    plt.show()
