import os
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms e Studi
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Algorithms.DQN import deepQlearning
from Studi.study_utils import plot_learning_curve

# Configurazione di base per lo studio
NUM_EPISODES = 20
SEEDS = (42, 123, 456)
ENV_NAME = "ALE/Freeway-v5"
OUTPUT_DIR = os.path.join("Studi", "output", "DQN")


def run_dqn_experiment(**kwargs):
    """
    Esegue l'algoritmo DQN con diversi semi e restituisce la matrice dei reward per episodio.
    """
    all_rewards = []
    for seed in seeds:
        print(f"  Esecuzione DQN con seed {seed} e parametri: {kwargs}")
        _, reward_history = deepQlearning(
            env_name=ENV_NAME,
            num_episodes=NUM_EPISODES,
            seed=seed,
            **kwargs
        )
        all_rewards.append(reward_history)
    return np.array(all_rewards)


def run_random_baseline():
    """
    Esegue una policy casuale di baseline con diversi semi.
    """
    all_rewards = []
    for seed in seeds:
        print(f"  Esecuzione Random Baseline con seed {seed}")
        env = gym.make(ENV_NAME)
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        reward_history = []
        for episode in range(NUM_EPISODES):
            obs, info = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            reward_history.append(total_reward)
        env.close()
        all_rewards.append(reward_history)
    return np.array(all_rewards)


def run_variants_comparison():
    """
    Confronta le 4 varianti DQN (ER + Target, ER, Target, Basic) e la baseline casuale.
    """
    print("\n=========================================")
    print("Confronto Varianti DQN e Baseline")
    print("=========================================")
    results = {}

    # 1) Full DQN (Experience Replay + Target Network)
    print("\n--- 1. Full DQN (ER + Target) ---")
    results["Full DQN (ER + Target)"] = run_dqn_experiment(
        use_experience_replay=True,
        use_target_network=True
    )

    # 2) Experience Replay Only
    print("\n--- 2. Experience Replay Only (No Target) ---")
    results["ER Only (No Target)"] = run_dqn_experiment(
        use_experience_replay=True,
        use_target_network=False
    )

    # 3) Target Network Only
    print("\n--- 3. Target Network Only (No ER) ---")
    results["Target Only (No ER)"] = run_dqn_experiment(
        use_experience_replay=False,
        use_target_network=True
    )

    # 4) Basic DQN
    print("\n--- 4. Basic DQN (No ER, No Target) ---")
    results["Basic DQN (No ER, No Target)"] = run_dqn_experiment(
        use_experience_replay=False,
        use_target_network=False
    )

    # 5) Baseline Casuale
    print("\n--- 5. Baseline Casuale ---")
    results["Random Baseline"] = run_random_baseline()

    # Genera grafico
    filename = os.path.join(OUTPUT_DIR, "dqn_variants_comparison.png")
    plot_learning_curve(
        results_dict=results,
        title=f"Confronto Varianti DQN e Baseline Casuale ({ENV_NAME})",
        filename=filename,
        window=1
    )
    print(f"\n✓ Grafico di confronto varianti salvato in: {filename}")


def run_epsilon_sensitivity():
    """
    Analisi di sensibilità ad epsilon.
    """
    print("\n=========================================")
    print("Analisi di Sensibilità a Epsilon (DQN Completo)")
    print("=========================================")
    results = {}
    
    epsilons = [0.01, 0.05, 0.1, 0.2]
    for eps in epsilons:
        print(f"\n--- Addestramento con Epsilon = {eps} ---")
        results[f"Epsilon = {eps}"] = run_dqn_experiment(
            use_experience_replay=True,
            use_target_network=True,
            epsilon=eps
        )

    # Genera grafico
    filename = os.path.join(OUTPUT_DIR, "dqn_epsilon_sensitivity.png")
    plot_learning_curve(
        results_dict=results,
        title=f"DQN — Sensibilità ad Epsilon ε ({ENV_NAME})",
        filename=filename,
        window=1
    )
    print(f"\n✓ Grafico di sensibilità epsilon salvato in: {filename}")


if __name__ == "__main__":
    # Configurazione semi fissi per lo studio
    seeds = SEEDS
    
    # Esegue il confronto tra varianti
    run_variants_comparison()
    
    # Esegue l'analisi di sensibilità
    run_epsilon_sensitivity()
    
    print(f"\n✓ Tutti gli studi completati! I grafici si trovano in: {OUTPUT_DIR}")
