"""Confronto tra algoritmi: stabilità e velocità di apprendimento in un unico grafico.

Esegue tutti e 4 gli algoritmi (Q-Learning, SARSA, Monte Carlo, TD) sullo stesso
ambiente e li confronta in un singolo plot che mostra:
- Learning curves sovrapposte (media ± std → stabilità)
- Marcatori sui punti in cui ciascun algoritmo raggiunge le soglie di reward (→ velocità)
- Grafico a barre separato per il confronto diretto della velocità
"""

import os
import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms e Studi
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Algorithms.QLearning import q_learning
from Algorithms.SARSA import sarsa
from Algorithms.MonteCarlo import montecarlo_prediction
from Algorithms.TemporalDifference import temporal_difference
from Studi.study_utils import (ENVIRONMENTS, run_experiment,
                                plot_comparison, plot_speed_of_learning,
                                plot_v_table_comparison)


NUM_EPISODES = 2000
SEEDS = (42, 123, 456)
OUTPUT_DIR = os.path.join("Studi", "output", "Confronto")

# Soglie per misurare la velocità di apprendimento
# Q-Learning e SARSA raggiungono reward alti; MC e TD usano altro
# quindi per il confronto usiamo soglie raggiungibili da tutti
SPEED_THRESHOLDS = {
    "CartPole": [20, 30, 50],
    "MountainCar": [-200, -195, -190],
}

# Iperparametri di default per ciascun algoritmo
ALGORITHM_CONFIGS = {
    "Q-Learning": {
        "fn": q_learning,
        "params": {"alpha": 0.2, "gamma": 0.99, "epsilon": 0.1},
    },
    "SARSA": {
        "fn": sarsa,
        "params": {"alpha": 0.2, "gamma": 0.99, "epsilon": 0.1},
    },
    "Monte Carlo": {
        "fn": montecarlo_prediction,
        "params": {"alpha": 0.1, "gamma": 0.99},
    },
    "TD(0)": {
        "fn": temporal_difference,
        "params": {"alpha": 0.1, "gamma": 0.99},
    },
}


def run_comparison(env_label="CartPole"):
    """Esegue tutti gli algoritmi e genera il grafico di confronto."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresholds = SPEED_THRESHOLDS[env_label]

    results = {}
    for algo_name, config in ALGORITHM_CONFIGS.items():
        print(f"\n  Running {algo_name} on {env_label}...")
        rewards = run_experiment(config["fn"], env["env_name"], env["bins_list"],
                                 num_episodes=NUM_EPISODES, seeds=SEEDS,
                                 **config["params"])
        results[algo_name] = rewards

    # 1) Grafico combinato: learning curves + marcatori di velocità
    print(f"\n=== Confronto Algoritmi — Stabilità e Velocità ({env_label}) ===")
    plot_comparison(
        results,
        title=f"Confronto Algoritmi — Stabilità e Velocità di Apprendimento ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"comparison_{env_label}.png"),
    )

    # 2) Grafico a barre: solo velocità di apprendimento
    plot_speed_of_learning(
        results,
        thresholds=thresholds,
        title=f"Confronto Algoritmi — Velocità di Apprendimento ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"speed_comparison_{env_label}.png"),
    )


def run_control_comparison(env_label="CartPole"):
    """Confronta solo gli algoritmi di controllo (Q-Learning vs SARSA)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Soglie più alte: gli algoritmi di controllo apprendono davvero
    control_thresholds = {
        "CartPole": [50, 100, 200],
        "MountainCar": [-180, -150, -130],
    }
    thresholds = control_thresholds[env_label]

    results = {}
    for algo_name in ["Q-Learning", "SARSA"]:
        config = ALGORITHM_CONFIGS[algo_name]
        print(f"\n  Running {algo_name} on {env_label}...")
        rewards = run_experiment(config["fn"], env["env_name"], env["bins_list"],
                                 num_episodes=NUM_EPISODES, seeds=SEEDS,
                                 **config["params"])
        results[algo_name] = rewards

    print(f"\n=== Q-Learning vs SARSA — Stabilità e Velocità ({env_label}) ===")
    plot_comparison(
        results,
        title=f"Q-Learning vs SARSA — Stabilità e Velocità ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"control_comparison_{env_label}.png"),
    )

    plot_speed_of_learning(
        results,
        thresholds=thresholds,
        title=f"Q-Learning vs SARSA — Velocità di Apprendimento ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"control_speed_{env_label}.png"),
    )


def run_prediction_comparison(env_label="CartPole"):
    """Confronta solo gli algoritmi di prediction (Monte Carlo vs TD)."""
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    thresholds = SPEED_THRESHOLDS[env_label]

    results = {}
    for algo_name in ["Monte Carlo", "TD(0)"]:
        config = ALGORITHM_CONFIGS[algo_name]
        print(f"\n  Running {algo_name} on {env_label}...")
        rewards = run_experiment(config["fn"], env["env_name"], env["bins_list"],
                                 num_episodes=NUM_EPISODES, seeds=SEEDS,
                                 **config["params"])
        results[algo_name] = rewards

    print(f"\n=== Monte Carlo vs TD(0) — Stabilità ({env_label}) ===")
    plot_comparison(
        results,
        title=f"Monte Carlo vs TD(0) — Stabilità ({env_label})",
        filename=os.path.join(OUTPUT_DIR, f"prediction_comparison_{env_label}.png"),
    )


def run_v_table_comparison(env_label="CartPole"):
    """Confronta i valori stimati nella V-table tra Monte Carlo e TD(0)."""
    import numpy as np
    env = ENVIRONMENTS[env_label]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n=== Confronto V-table: Monte Carlo vs TD(0) ({env_label}) ===")
    
    # Eseguiamo MC
    print("  Addestramento Monte Carlo...")
    ## Il seed serve solo per la posizione iniziale
    np.random.seed(42)
    mc_v_table, _ = ALGORITHM_CONFIGS["Monte Carlo"]["fn"](
        env["env_name"], env["bins_list"], num_episodes=NUM_EPISODES, 
        **ALGORITHM_CONFIGS["Monte Carlo"]["params"]
    )
    
    # Eseguiamo TD
    print("  Addestramento TD(0)...")
    np.random.seed(42)
    td_v_table, _ = ALGORITHM_CONFIGS["TD(0)"]["fn"](
        env["env_name"], env["bins_list"], num_episodes=NUM_EPISODES, 
        **ALGORITHM_CONFIGS["TD(0)"]["params"]
    )
    
    # Confronto
    common_states = set(mc_v_table.keys()).intersection(set(td_v_table.keys()))
    if not common_states:
        print("  Nessuno stato in comune visitato!")
        return
        
    differences = []
    for state in common_states:
        differences.append(abs(mc_v_table[state] - td_v_table[state]))
        
    mse = np.mean(np.square(differences))
    mae = np.mean(differences)
    
    print(f"  Stati in comune valutati: {len(common_states)}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    
    # Salviamo un report testuale
    report_file = os.path.join(OUTPUT_DIR, f"v_table_comparison_{env_label}.txt")
    with open(report_file, "w") as f:
        f.write(f"Confronto V-table: Monte Carlo vs TD(0) ({env_label})\n")
        f.write(f"Stati in comune valutati: {len(common_states)}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    print(f"✓ Report salvato in: {report_file}")
    
    # Salviamo il grafico con le heatmaps di confronto delle V-table
    heatmap_file = os.path.join(OUTPUT_DIR, f"v_table_heatmap_{env_label}.png")
    plot_v_table_comparison(mc_v_table, td_v_table, env_label, env["bins_list"], heatmap_file)


if __name__ == "__main__":
    for env_label in ["CartPole", "MountainCar"]:
        # Confronto globale (tutti e 4)
        run_comparison(env_label)
        # Confronto controllo (Q-Learning vs SARSA)
        run_control_comparison(env_label)
        # Confronto prediction (MC vs TD)
        run_prediction_comparison(env_label)
        # Confronto tabelle V (MC vs TD)
        run_v_table_comparison(env_label)

    print("\n✓ Tutti i confronti completati. Grafici in:", OUTPUT_DIR)
