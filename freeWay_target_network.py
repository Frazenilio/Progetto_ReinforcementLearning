import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
# 0 NOP,1 UP , 2 DOWN
ACTIONS = [0,1,2]

class FreewayCNN(nn.Module):
    """
    CNN per il processing di frame Atari RGB (210, 160, 3).
    
    Architettura:
        - 3 layer convoluzionali (con BatchNorm, ReLU, MaxPool)
        - Rete feedforward finale → output di 3 elementi
    
    Input atteso dalla forward: tensor di shape (batch, 3, 210, 160)
    """

    def __init__(self):
        super(FreewayCNN, self).__init__()

        # --- Layer Convoluzionali ---
        # Conv1: 3 canali RGB → 32 filtri, kernel 8x8, stride 4
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv2: 32 → 64 filtri, kernel 4x4, stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv3: 64 → 64 filtri, kernel 3x3, stride 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # --- Calcolo dimensione dopo le convoluzioni ---
        # Per determinare la dimensione del flatten, passiamo un tensor fittizio
        dummy = torch.zeros(1, 3, 210, 160)
        dummy = self._conv_forward(dummy)
        flatten_size = dummy.view(1, -1).shape[1]

        # --- Rete Feedforward ---
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 3)  # output: 3 valori

    def _conv_forward(self, x):
        """Passa x attraverso i 3 layer convoluzionali."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        """
        Forward pass completo.
        
        Args:
            x: tensor di shape (batch, 3, 210, 160), valori in [0, 1]
        
        Returns:
            tensor di shape (batch, 3)
        """
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def compute_loss(self, q_value, action, reward, next_q_value, done, gamma):
        """
        Loss DQN: L(w) = 1/2 * (R_{t+1} + γ * max_a q_w(S_{t+1}, a) - q_w(S_t, A_t))^2
        
        Il target (R + γ * max Q') è detachato dal grafo: semi-gradient update.
        
        Args:
            q_value:      tensor (1, 3) — Q-values dello stato corrente
            action:       int — azione eseguita
            reward:       float — reward ricevuto
            next_q_value: tensor (1, 3) — Q-values dello stato successivo
            done:         bool — True se l'episodio è terminato
            gamma:        float — fattore di sconto
        
        Returns:
            loss: tensor scalare
        """
        # q_w(S_t, A_t): Q-value dell'azione scelta
        q_sa = q_value[0, action]

        # Target: R_{t+1} + γ * max_a q_w(S_{t+1}, a)
        # Se done, il target è solo R (non c'è stato futuro)
        with torch.no_grad():
            max_next_q = torch.max(next_q_value)
            target = reward + gamma * max_next_q * (1 - int(done))

        # L(w) = 1/2 * (target - q_w(S_t, A_t))^2
        loss = 0.5 * (target - q_sa) ** 2
        return loss

    @staticmethod
    def preprocess(obs):
        """
        Converte un'osservazione numpy (210, 160, 3) uint8 in un tensor
        PyTorch (1, 3, 210, 160) float32 normalizzato in [0, 1].
        
        Args:
            obs: numpy array di shape (210, 160, 3), dtype uint8
        
        Returns:
            tensor di shape (1, 3, 210, 160), dtype float32
        """
        # (H, W, C) → (C, H, W), normalizza a [0,1], aggiungi dimensione batch
        tensor = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)  # (1, 3, 210, 160)

def deepQlearning(env_name: str, num_episodes: int = 5000,update_step: int=20, render: bool = False,
               lr: float = 1e-4, gamma: float = 0.99,
               epsilon: float = 0.1, epsilon_min: float = 0.01,
               epsilon_decay: float = 0.999):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Inizializza la rete e l'ottimizzatore
    dqn = FreewayCNN()
    target_dqn = copy.deepcopy(dqn)
    optimizer = torch.optim.SGD(dqn.parameters(), lr=lr)
    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        count_step=0
        while not done:
            # Preprocessing: numpy (210,160,3) → tensor (1,3,210,160)
            preprocessed_obs = FreewayCNN.preprocess(obs)

            # Forward pass: ottieni Q-values per tutte le azioni
            q_value = dqn(preprocessed_obs)  # shape (1, 3)

            # Epsilon-greedy: esplora con probabilità epsilon
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # esplorazione
            else:
                # Sfruttamento: scegli l'azione con Q-value massimo
                action = ACTIONS[torch.argmax(q_value).item()]

            # Esegui l'azione nell'ambiente
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            count_step+=1
            if count_step==update_step:
                target_dqn = copy.deepcopy(dqn)
                count_step=0

            # Forward pass sullo stato successivo
            preprocessed_next_obs = FreewayCNN.preprocess(next_obs)
            next_q_value = target_dqn(preprocessed_next_obs)  # shape (1, 3)

            # Calcola la loss DQN:
            # L(w) = 1/2 * (R + γ * max_a q(S',a) - q(S, A))^2
            loss = dqn.compute_loss(q_value, action, reward, next_q_value, done, gamma)

            # Backpropagation e aggiornamento pesi
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            episode_reward += reward

        # Decay epsilon dopo ogni episodio
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        # Stampa progresso ogni 100 episodi
        #if (episode + 1) % 100 == 0:
        avg_reward = np.mean(reward_history[-100:])
        print(f"Episodio {episode + 1}/{num_episodes} | "
                f"Reward medio (ultimi 100): {avg_reward:.1f} | "
                f"Epsilon: {epsilon:.3f}")

    env.close()
    return dqn, reward_history



if __name__ == "__main__":
    # Training Q-learning su CartPole con discretizzazione
    deepQlearning("ALE/Freeway-v5", num_episodes=5000, render=True)