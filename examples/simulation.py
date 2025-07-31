from __future__ import annotations

from sulfur_simulation.scattering_calculation import SimulationParameters

if __name__ == "__main__":
    # Create a simulation with 5000 timesteps
    params = SimulationParameters(delta=1.2, duration=5000, gamma=1, step=2)
