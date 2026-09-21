"""Studio Q-Learning: learning curve, velocità di apprendimento e sensibilità (alpha, gamma, epsilon)."""

import os
import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms e Studi
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Algorithms.QLearning import q_learning
from Studi.study_utils import (ENVIRONMENTS, run_experiment,
                                plot_learning_curve, plot_sensitivity,
                                plot_speed_of_learning)

# Soglie di reward per misurare la velocità di apprendimento
SPEED_THRESHOLDS = {
    "CartPole": [50, 100, 200],
    "MountainCar": [-180, -150, -130],
}

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
NUM_EPISODES = 2000
SEEDS = (42, 123, 456)
OUTPUT_DIR = os.path.join("Studi", "output", "QLearning")


def run_learning_curve(env_label="CartPole"):
    """Genera la learning curve di Q-Learning su un ambiente."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Q-Learning — Learning Curve ({env_label}) ===")
    rewards = run_experiment(q_learning, env["env_name"], env["bins_list"],
                             num_episodes=NUM_EPISODES, seeds=SEEDS,
                             alpha=0.2, gamma=0.99, epsilon=0.1)

    plot_learning_curve(
        {f"Q-Learning": rewards},
        title=f"Q-Learning — Learning Curve ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"learning_curve_{env_label}.png"),
    )


def run_sensitivity_alpha(env_label="CartPole"):
    """Analisi di sensibilità al learning rate (alpha)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Q-Learning — Sensibilità Alpha ({env_label}) ===")
    plot_sensitivity(
        algorithm_fn=q_learning,
        env_name=env["env_name"],
        bins_list=env["bins_list"],
        param_name="alpha",
        param_values=[0.05, 0.1, 0.2, 0.5],
        fixed_params={"gamma": 0.99, "epsilon": 0.1},
        title=f"Q-Learning — Sensibilità al Learning Rate α ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"sensitivity_alpha_{env_label}.png"),
        num_episodes=NUM_EPISODES,
        seeds=SEEDS,
    )


def run_sensitivity_gamma(env_label="CartPole"):
    """Analisi di sensibilità al fattore di sconto (gamma)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Q-Learning — Sensibilità Gamma ({env_label}) ===")
    plot_sensitivity(
        algorithm_fn=q_learning,
        env_name=env["env_name"],
        bins_list=env["bins_list"],
        param_name="gamma",
        param_values=[0.9, 0.95, 0.99, 1.0],
        fixed_params={"alpha": 0.2, "epsilon": 0.1},
        title=f"Q-Learning — Sensibilità al Fattore di Sconto γ ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"sensitivity_gamma_{env_label}.png"),
        num_episodes=NUM_EPISODES,
        seeds=SEEDS,
    )


def run_sensitivity_epsilon(env_label="CartPole"):
    """Analisi di sensibilità alla probabilità di esplorazione (epsilon)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Q-Learning — Sensibilità Epsilon ({env_label}) ===")
    plot_sensitivity(
        algorithm_fn=q_learning,
        env_name=env["env_name"],
        bins_list=env["bins_list"],
        param_name="epsilon",
        param_values=[0.01, 0.05, 0.1, 0.3],
        fixed_params={"alpha": 0.2, "gamma": 0.99},
        title=f"Q-Learning — Sensibilità a Epsilon ε ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"sensitivity_epsilon_{env_label}.png"),
        num_episodes=NUM_EPISODES,
        seeds=SEEDS,
    )


def run_speed_of_learning(env_label="CartPole"):
    """Analisi della velocità di apprendimento (episodi per raggiungere soglie di reward)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresholds = SPEED_THRESHOLDS[env_label]

    print(f"\n=== Q-Learning — Velocità di Apprendimento ({env_label}) ===")
    rewards = run_experiment(q_learning, env["env_name"], env["bins_list"],
                             num_episodes=NUM_EPISODES, seeds=SEEDS,
                             alpha=0.2, gamma=0.99, epsilon=0.1)

    plot_speed_of_learning(
        {"Q-Learning": rewards},
        thresholds=thresholds,
        title=f"Q-Learning — Velocità di Apprendimento ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"speed_of_learning_{env_label}.png"),
    )


if __name__ == "__main__":
    for env_label in ["CartPole", "MountainCar"]:
        run_learning_curve(env_label)
        run_speed_of_learning(env_label)
        run_sensitivity_alpha(env_label)
        run_sensitivity_gamma(env_label)
        run_sensitivity_epsilon(env_label)

    print("\n✓ Tutti gli studi Q-Learning completati. Grafici in:", OUTPUT_DIR)
