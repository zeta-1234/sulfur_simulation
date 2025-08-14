"""Example simulation for a square lattice with a Lennard Jones interacting potential."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hopping_calculator import (
    InteractingHoppingCalculator,
    get_lennard_jones_potential,
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
    jump_counter,
    run_simulation,
    sampled_jumps,
)
from sulfur_simulation.show_simulation import (
    animate_particle_positions_hexagonal,
    animate_particle_positions_square,
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=100,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=InteractingHoppingCalculator(
            baserate=0.01,
            temperature=200,
            lattice_spacing=2.5,
            interaction=get_lennard_jones_potential(sigma=2.55, epsilon=0.03 * 1.6e-19),
        ),
    )

    positions = run_simulation(params=params)
    isf_params = ISFParameters(params=params)
    amplitudes = get_amplitudes(isf_params=isf_params, positions=positions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_isf(
        x=amplitudes, t=params.times, ax=ax1, delta_k_index=50, isf_params=isf_params
    )

    dephasing_rates = get_dephasing_rates(amplitudes=amplitudes, t=params.times)

    plot_dephasing_rates(
        dephasing_rates=dephasing_rates,
        delta_k=isf_params.delta_k_array[:, 0],
        ax=ax2,
    )

    timesteps = np.arange(1, 100)[::1]

    anim = animate_particle_positions_square(
        all_positions=positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    anim2 = animate_particle_positions_hexagonal(
        all_positions=positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    plt.show()
    print(f"sampled jumps: {sampled_jumps}")
    print(jump_counter)

    indices = np.arange(len(jump_counter))
    width = 0.35  # width of the bars

    plt.bar(indices - width / 2, jump_counter, width, label="jump_counter")
    plt.bar(indices + width / 2, sampled_jumps, width, label="sampled_jumps")
    plt.xlabel("Jump index")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

# TODO: fit ISFs better, try a fixed window
# TODO: longer simulation, improve efficiency
# TODO: Gillespie algorithm??
# TODO: decrease base jump rate
