import sys
from pathlib import Path

# Aggiunge la cartella radice del progetto al sys.path per consentire gli import di Algorithms
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Algorithms.DQN import deepQlearning

ENV_NAME = "ALE/Freeway-v5"

if __name__ == "__main__":
    # Addestra un agente DQN su Freeway-v5 usando le impostazioni di default
    # (sia Experience Replay che Target Network abilitati)
    print("Avvio addestramento DQN su Freeway...")
    dqn_model, rewards = deepQlearning(
        env_name=ENV_NAME,
        num_episodes=5,  # Numero ridotto per test rapido
        render=False,
        epsilon=0.1,
        use_experience_replay=True,
        use_target_network=True,
        target_update_step=20,
        seed=42
    )
    print(f"Addestramento completato! Storico reward degli episodi: {rewards}")
