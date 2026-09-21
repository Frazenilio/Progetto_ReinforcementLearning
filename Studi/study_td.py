"""Studio TD(0) Prediction: learning curve, velocità di apprendimento e sensibilità (alpha, gamma).

Nota: TD prediction usa una policy random, quindi non ha il parametro epsilon.
"""

import os
import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms e Studi
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Algorithms.TemporalDifference import temporal_difference
from Studi.study_utils import (ENVIRONMENTS, run_experiment,
                                plot_learning_curve, plot_sensitivity,
                                plot_speed_of_learning)

SPEED_THRESHOLDS = {
    "CartPole": [15, 20, 25],
    "MountainCar": [-200, -195, -190],
}

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
NUM_EPISODES = 2000
SEEDS = (42, 123, 456)
OUTPUT_DIR = os.path.join("Studi", "output", "TD")


def run_learning_curve(env_label="CartPole"):
    """Genera la learning curve di TD(0) prediction su un ambiente."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== TD(0) — Learning Curve ({env_label}) ===")
    rewards = run_experiment(temporal_difference, env["env_name"], env["bins_list"],
                             num_episodes=NUM_EPISODES, seeds=SEEDS,
                             alpha=0.1, gamma=0.99)

    plot_learning_curve(
        {f"TD(0)": rewards},
        title=f"TD(0) Prediction — Learning Curve ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"learning_curve_{env_label}.png"),
    )


def run_sensitivity_alpha(env_label="CartPole"):
    """Analisi di sensibilità al learning rate (alpha)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== TD(0) — Sensibilità Alpha ({env_label}) ===")
    plot_sensitivity(
        algorithm_fn=temporal_difference,
        env_name=env["env_name"],
        bins_list=env["bins_list"],
        param_name="alpha",
        param_values=[0.01, 0.05, 0.1, 0.3],
        fixed_params={"gamma": 0.99},
        title=f"TD(0) — Sensibilità al Learning Rate α ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"sensitivity_alpha_{env_label}.png"),
        num_episodes=NUM_EPISODES,
        seeds=SEEDS,
    )


def run_sensitivity_gamma(env_label="CartPole"):
    """Analisi di sensibilità al fattore di sconto (gamma)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== TD(0) — Sensibilità Gamma ({env_label}) ===")
    plot_sensitivity(
        algorithm_fn=temporal_difference,
        env_name=env["env_name"],
        bins_list=env["bins_list"],
        param_name="gamma",
        param_values=[0.9, 0.95, 0.99, 1.0],
        fixed_params={"alpha": 0.1},
        title=f"TD(0) — Sensibilità al Fattore di Sconto γ ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"sensitivity_gamma_{env_label}.png"),
        num_episodes=NUM_EPISODES,
        seeds=SEEDS,
    )



def run_speed_of_learning(env_label="CartPole"):
    """Analisi della velocità di apprendimento."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresholds = SPEED_THRESHOLDS[env_label]

    print(f"\n=== TD(0) — Velocità di Apprendimento ({env_label}) ===")
    rewards = run_experiment(temporal_difference, env["env_name"], env["bins_list"],
                             num_episodes=NUM_EPISODES, seeds=SEEDS,
                             alpha=0.1, gamma=0.99)

    plot_speed_of_learning(
        {"TD(0)": rewards},
        thresholds=thresholds,
        title=f"TD(0) — Velocità di Apprendimento ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"speed_of_learning_{env_label}.png"),
    )


if __name__ == "__main__":
    for env_label in ["CartPole", "MountainCar"]:
        run_learning_curve(env_label)
        run_speed_of_learning(env_label)
        run_sensitivity_alpha(env_label)
        run_sensitivity_gamma(env_label)

    print("\n✓ Tutti gli studi TD(0) completati. Grafici in:", OUTPUT_DIR)
