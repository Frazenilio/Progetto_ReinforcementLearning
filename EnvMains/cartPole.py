import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from Algorithms.QLearning import q_learning
from Algorithms.SARSA import sarsa
from Algorithms.MonteCarlo import montecarlo_prediction
from Algorithms.TemporalDifference import temporal_difference

MAX_VELOCITY = 3.5
MAX_ANGULAR_VELOCITY = 3.5
MAX_POSITION = 2.4
MAX_ANGLE = 0.21
MIN_VELOCITY = -3.5
MIN_ANGULAR_VELOCITY = -3.5
MIN_POSITION = -2.4
MIN_ANGLE = -0.21

ENV_NAME = "CartPole-v1"


def make_bins(num_positions=24, num_angles=30,
              num_velocities=24, num_angular_velocities=24):
    """Crea la lista di bin per discretizzare lo spazio degli stati di CartPole.

    L'ordine dei bin corrisponde all'ordine dell'osservazione Gymnasium:
    [position, velocity, angle, angular_velocity].

    Ritorna
    -------
    list[np.ndarray]
        Lista di 4 array: positions, velocities, angles, angular_velocities.
    """
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    angles = np.linspace(MIN_ANGLE, MAX_ANGLE, num_angles)
    angular_velocities = np.linspace(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY,
                                     num_angular_velocities)
    # Ordine Gymnasium: pos, vel, ang, ang_vel
    return [positions, velocities, angles, angular_velocities]


if __name__ == "__main__":
    bins_list = make_bins()

    # Scegli l'algoritmo da eseguire decommentando la riga desiderata:
    # q_table, rewards = q_learning(ENV_NAME, bins_list, num_episodes=5000)
    # q_table, rewards = sarsa(ENV_NAME, bins_list, num_episodes=5000)
    # v_table, rewards = montecarlo_prediction(ENV_NAME, bins_list, num_episodes=5000)
    v_table, rewards = temporal_difference(ENV_NAME, bins_list, num_episodes=5000)