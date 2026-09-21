import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Algorithms.QLearning import q_learning
from Algorithms.SARSA import sarsa
from Algorithms.MonteCarlo import montecarlo_prediction
from Algorithms.TemporalDifference import temporal_difference

# MountainCar-v0: osservazione = [position, velocity]
# Position: [-1.2, 0.6]  (goal >= 0.5)
# Velocity: [-0.07, 0.07]
MAX_VELOCITY = 0.07
MAX_POSITION = 0.6
MIN_VELOCITY = -0.07
MIN_POSITION = -1.2

ENV_NAME = "MountainCar-v0"


def make_bins(num_positions=24, num_velocities=24):
    """Crea la lista di bin per discretizzare lo spazio degli stati di MountainCar.

    L'ordine dei bin corrisponde all'ordine dell'osservazione Gymnasium:
    [position, velocity].

    Ritorna
    -------
    list[np.ndarray]
        Lista di 2 array: positions, velocities.
    """
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    return [positions, velocities]


if __name__ == "__main__":
    bins_list = make_bins()

    # Scegli l'algoritmo da eseguire decommentando la riga desiderata:
    # q_table, rewards = q_learning(ENV_NAME, bins_list, num_episodes=5000)
    q_table, rewards = sarsa(ENV_NAME, bins_list, num_episodes=5000)
    # v_table, rewards = montecarlo_prediction(ENV_NAME, bins_list, num_episodes=5000)
    # v_table, rewards = temporal_difference(ENV_NAME, bins_list, num_episodes=5000)