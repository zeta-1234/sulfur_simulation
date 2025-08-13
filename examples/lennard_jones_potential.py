"""Example simulation for a square lattice with a Lennard Jones interacting potential."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from sulfur_simulation.hopping_calculator import (InteractingParticlesHoppingCalculator, get_lennard_jones_potential)
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
)

if __name__ == "__main__":
    params = SimulationParameters(
        n_timesteps=12000,
        lattice_dimension=(100, 100),
        n_particles=500,
        hopping_calculator=InteractingParticlesHoppingCalculator(
            baserate=0.01,
            temperature=200,
            lattice_spacing=2.5,
            interaction=get_lennard_jones_potential(
                sigma=2.55, 
                epsilon=0.03 * 1.6e-19,
            )
    ))

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

    timesteps = np.arange(1, 12000)[::20]

    anim = animate_particle_positions(
        all_positions=positions,
        lattice_dimension=params.lattice_dimension,
        timesteps=timesteps,
        lattice_spacing=2.5,
    )

    plt.show()
