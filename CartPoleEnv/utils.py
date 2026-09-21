import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class Discretizer:
    def __init__(self, n_bins=(6, 12, 12, 24)):
        """
        Discretizer per mappare lo spazio di stato continuo di CartPole in uno discreto.
        n_bins: numero di bin per (position, velocity, angle, angular_velocity)
        """
        self.n_bins = n_bins
        # Limiti ragionevoli per le variabili (position, velocity, angle, angular_velocity)
        # L'ambiente termina se la posizione > 2.4 o <-2.4, e angolo > 0.209 (12 deg) o <-0.209
        self.lower_bounds = [-2.4, -3.0, -0.209, -3.0]
        self.upper_bounds = [2.4, 3.0, 0.209, 3.0]
        
    def discretize(self, obs):
        discretized = []
        for i in range(len(obs)):
            # Normalizza tra 0 e 1
            scaled = (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            # Vincola ai limiti
            scaled = max(0.0, min(1.0, scaled))
            # Trova l'indice del bin
            bin_idx = int(np.round(scaled * (self.n_bins[i] - 1)))
            discretized.append(bin_idx)
        return tuple(discretized)

def plot_learning_curves(results, title, filename):
    """
    results: dict formato {'Label': array di shape (n_seeds, n_episodes)}
    """
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)
        
        # Smoothing (moving average) per una migliore visualizzazione
        window = max(int(len(mean_rewards) * 0.05), 1)
        mean_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        std_rewards = np.convolve(std_rewards, np.ones(window)/window, mode='valid')
        
        episodes = np.arange(len(mean_rewards))
        
        plt.plot(episodes, mean_rewards, label=label)
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
        
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_v_values(v_dict, title, filename):
    values = list(v_dict.values())
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50)
    plt.title(title)
    plt.xlabel('State Value V(s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
