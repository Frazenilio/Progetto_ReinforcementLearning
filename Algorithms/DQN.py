import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque

# 0: NOP, 1: UP, 2: DOWN
ACTIONS = [0, 1, 2]

class ReplayBuffer:
    def __init__(self, capacity=10000):
        ## Prepara il buffer per ER
        self.buffer = deque(maxlen=capacity)

    ## Aggiunge una transazione al buffer dell'ER. Gli stati sono numpy array (210, 160, 3) uint8
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    ## Campiona (ritorna) casualmente delle tuple dal buffer di ER e le converte in tensor
    def sample(self, batch_size, device="cpu"):
        batch = random.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        ## Converte le immagini numpy in tensor float32 sul device corretto on-the-fly
        states = torch.cat([FreewayCNN.preprocess(s, device) for s in states], dim=0)
        next_states = torch.cat([FreewayCNN.preprocess(ns, device) for ns in next_states], dim=0)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

## CNN per il processing di frame Atari RGB (210, 160, 3).
## Ha 3 layer convoluzionali ognuno con BatchNorm e ReLU
class FreewayCNN(nn.Module):
    def __init__(self):
        super(FreewayCNN, self).__init__()

        ## Conv1: 210x160@3 -> floor( (210-8)/4 + 1 x (160-8)/4 + 1 ) @ 32 -> 51x39@32 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        # 51x39@32 -> floor( (51-4)/2 ) + 1 x floor( (39-4)/2 ) + 1 @ 64 -> 24x18@64 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        # 24x18@64 -> floor( (24-3)/1 ) + 1 x floor( (18-3)/1 ) + 1 @ 64 -> 22x16@64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        flatten_size = 64 * 22 * 16

        # --- Rete Feedforward ---
        self.fc1 = nn.Linear(flatten_size, 512)
        self.fc2 = nn.Linear(512, 3)  # output: 3 valori

    def _conv_forward(self, x):
        """Passa x attraverso i 3 layer convoluzionali."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    ## Forward da usare
    def forward(self, x):
        x = self._conv_forward(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    ## La loss qui e' differente, controllare le slide (o la relazione) per la formula
    def compute_loss(self, q_value, action, reward, next_q_value, done, gamma):
        q_sa = q_value[0, action]
        with torch.no_grad():
            max_next_q = torch.max(next_q_value)
            target = reward + gamma * max_next_q * (1 - int(done))
        loss = 0.5 * (target - q_sa) ** 2
        return loss

    ## Come la loss sopra ma per batch (infatti si fa la media)
    ## con questo si può usare la target_net per di più
    def compute_loss_batch(self, states, actions, rewards, next_states, dones, gamma, target_net=None):

        ## Si ripassano gli stati samplati
        q_values = self.forward(states)

        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (N,)

        with torch.no_grad():
            net_for_target = target_net if target_net is not None else self
            next_q_values = net_for_target(next_states)
            max_next_q, _ = torch.max(next_q_values, dim=1)
            targets = rewards + gamma * max_next_q * (1 - dones)

        ## Media delle loss per la loss finale
        loss = 0.5 * torch.mean((targets - q_sa) ** 2)
        return loss

    @staticmethod
    def preprocess(obs, device="cpu"):
        """
        Converte un'osservazione numpy (210, 160, 3) uint8 in un tensor
        PyTorch (1, 3, 210, 160) float32 normalizzato in [0, 1], caricato sul device.
        """
        tensor = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(device)  # (1, 3, 210, 160)


def deepQlearning(env_name: str, num_episodes: int = 5000, render: bool = False,
                  lr: float = 1e-4, gamma: float = 0.99, epsilon: float = 0.1,
                  use_experience_replay: bool = True, use_target_network: bool = True,
                  buffer_size: int = 10000, batch_size: int = 32,
                  target_update_step: int = 20, seed: int = None):
    """
    Addestra un agente DQN sull'ambiente Freeway con opzioni per Experience Replay e Target Network.
    
    Parametri
    ---------
    env_name : str
        Nome dell'ambiente Atari (es. "ALE/Freeway-v5").
    num_episodes : int
        Numero di episodi di training.
    render : bool
        Se True, mostra la visualizzazione grafica.
    lr : float
        Learning rate dell'ottimizzatore SGD.
    gamma : float
        Fattore di sconto.
    epsilon : float
        Probabilità di esplorazione epsilon-greedy (fissa).
    use_experience_replay : bool
        Se True, usa il buffer per aggiornamenti a batch.
    use_target_network : bool
        Se True, stabilizza il target con una target network separata.
    buffer_size : int
        Capacità del replay buffer (numero massimo di transizioni memorizzate).
    batch_size : int
        Dimensione del mini-batch campionato casualmente dal replay buffer.
    target_update_step : int
        Frequenza di aggiornamento della target network (in step totali).
    seed : int, optional
        Seme di randomizzazione per la riproducibilità.
    
    Ritorna
    -------
    dqn : FreewayCNN
        Rete neurale addestrata.
    reward_history : list[float]
        Storico dei reward totali ottenuti per episodio.
    """
    ## Seed set
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Rilevamento automatico del device (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DQN in esecuzione su device: {device} | ER: {use_experience_replay} | TargetNet: {use_target_network}")

    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    dqn = FreewayCNN().to(device)
    optimizer = torch.optim.SGD(dqn.parameters(), lr=lr)
    
    ## Inizializzamo subito la TN se serve
    target_dqn = None
    if use_target_network:
        target_dqn = copy.deepcopy(dqn).to(device)

    ## Idem per il buffer di ER
    replay_buffer = None
    if use_experience_replay:
        replay_buffer = ReplayBuffer(capacity=buffer_size)

    reward_history = []

    for episode in range(num_episodes):
        # Reset dell'ambiente con seed appropriato
        if seed is not None:
            obs, info = env.reset(seed=seed + episode)
        else:
            obs, info = env.reset()

        done = False
        episode_reward = 0.0
        count_step = 0

        while not done:
            ## Il preprocessing serve per avere osservazioni come tensori
            preprocessed_obs = FreewayCNN.preprocess(obs, device)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # esplorazione
            else:
                ## Apparentemente, senza il torch no grad, pytorch dovrebbe ricordare questo conto
                ## cosa che non serve quindi mettiamo il no grad
                with torch.no_grad():
                    q_value = dqn(preprocessed_obs)
                action = ACTIONS[torch.argmax(q_value).item()]  # sfruttamento

            ## Esegui l'azione
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            count_step += 1

            ## Se si usa la TN e abbiamo raggiunto l'update step, aggiorniamo la TN
            ## Questo dovrebbe rompere la Deadly Triad
            if use_target_network and count_step % target_update_step == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            ## Se usiamo ER, addestriamo samplando (se ci sono abbastanza episodi passati)
            if use_experience_replay:
                ## Salva la transizione nel buffer (salviamo l'osservazione grezza uint8)
                replay_buffer.push(obs, action, reward, next_obs, done)

                # Aggiorna se il buffer ha abbastanza transizioni per un mini-batch
                if len(replay_buffer) >= batch_size:
                    states, actions_batch, rewards_batch, next_states, dones_batch = replay_buffer.sample(batch_size, device)

                    loss = dqn.compute_loss_batch(
                        states, actions_batch, rewards_batch, next_states, dones_batch, gamma,
                        target_net=(target_dqn if use_target_network else None)
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            ## Se invece non si usa ER, facciamo un update "normale"
            else:
                preprocessed_next_obs = FreewayCNN.preprocess(next_obs, device)
                states = preprocessed_obs
                next_states = preprocessed_next_obs
                actions_batch = torch.tensor([action], dtype=torch.long, device=device)
                rewards_batch = torch.tensor([reward], dtype=torch.float32, device=device)
                dones_batch = torch.tensor([float(done)], dtype=torch.float32, device=device)

                loss = dqn.compute_loss_batch(
                    states, actions_batch, rewards_batch, next_states, dones_batch, gamma,
                    target_net=(target_dqn if use_target_network else None)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            obs = next_obs
            episode_reward += reward

        reward_history.append(episode_reward)

        ## per controllare, si puo' stampare il progresso ad ogni episodio
        # avg_reward = np.mean(reward_history[-100:]) if len(reward_history) > 0 else episode_reward
        # print(f"Episodio {episode + 1}/{num_episodes} | "
        #       f"Reward totale: {episode_reward:.1f} | "
        #       f"Reward medio (ultimi 100): {avg_reward:.1f} | "
        #       f"Epsilon: {epsilon:.3f}")

    env.close()
    return dqn, reward_history
