from __future__ import annotations

import numpy as np

from sulfur_simulation.scattering_calculation import SimulationParameters

if __name__ == "__main__":
    # Create a simulation with 5000 timesteps
    params = SimulationParameters(
        total_scattering_angle=1.2,
        angle_of_incidence=0.8,
        n_timesteps=5000,
        step=2,
        incident_wavevector=np.array([3, 6]),
    )
