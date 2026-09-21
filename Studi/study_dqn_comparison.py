import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms e Studi
sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from Algorithms.DQN import dqn

def main():
    # Usiamo CartPole-v1 per uno studio di ablazione rapido (più veloce di Freeway)
    env_name = "CartPole-v1"
    print(f"Inizializzazione dell'ambiente per lo studio di ablazione DQN: {env_name}...")

    try:
        env = gym.make(env_name)
    except Exception as e:
        print(f"Errore nella creazione dell'ambiente {env_name}: {e}")
        sys.exit(1)

    num_episodes = 150

    # 4 Configurazioni di ablazione da testare e confrontare
    configs = {
        "DQN Completo (Replay + Target)": {"use_replay_buffer": True, "use_target_network": True},
        "DQN senza Target Network":       {"use_replay_buffer": True, "use_target_network": False},
        "DQN senza Experience Replay":   {"use_replay_buffer": False, "use_target_network": True},
        "DQN senza entrambi":             {"use_replay_buffer": False, "use_target_network": False}
    }

    results = {}

    for label, flags in configs.items():
        print(f"\n==========================================")
        print(f"Avvio addestramento: {label}")
        print(f"==========================================")

        # Ripristino seeds per un confronto equo
        np.random.seed(42)
        import torch
        torch.manual_seed(42)
        env.reset(seed=42)

        # Eseguiamo il training
        _, reward_history = dqn(
            env,
            num_episodes=num_episodes,
            batch_size=64,
            lr=1e-3,  # CartPole si adatta bene ad un LR leggermente più alto
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.98,
            target_update_freq=500,
            buffer_size=20000,
            minimal_buffer_size=500,
            use_soft_update=True,
            tau=0.01,
            **flags
        )
        results[label] = reward_history

    env.close()

    # Creazione del grafico comparativo
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'orange', 'green', 'red']

    for i, (label, rewards) in enumerate(results.items()):
        # Sfondo con la curva grezza opaca
        plt.plot(rewards, color=colors[i], alpha=0.15)

        # Media mobile (finestra 10 ep) per pulire il grafico
        window = 10
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            plt.plot(np.arange(window-1, len(rewards)), smoothed,
                     label=label, color=colors[i], linewidth=2.5)
        else:
            plt.plot(rewards, label=label, color=colors[i], linewidth=2.5)

    plt.title(f"Studio di Ablazione DQN su {env_name} (Impatto di Replay e Target Net)", fontsize=14, fontweight="bold")
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel("Reward Totale (Media Mobile)", fontsize=12)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)

    # Directory di output
    output_dir = os.path.join("Studi", "output", "DQN")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, "dqn_ablation_study.png")
    try:
        plt.savefig(plot_path, dpi=150)
        print(f"\nGrafico dello studio di ablazione salvato in: {plot_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio del grafico: {e}")

    plt.close()

if __name__ == "__main__":
    main()
