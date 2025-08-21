"""Example simulation for a square lattice with a potential energy well across the center."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
    animate_particle_positions_square,
)
from sulfur_simulation.square_hopping_calculator import (
    LineDefectHoppingCalculator,
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=12000,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=LineDefectHoppingCalculator(
            straight_baserate=0.01,
            diagonal_baserate=0.01 / 5,
            temperature=200,
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

    timesteps = np.arange(1, 12001)[::100]

    anim = animate_particle_positions_square(
        all_positions=results[0].positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    plt.show()
