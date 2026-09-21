import numpy as np
from collections import defaultdict

class MCPrediction:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.V = defaultdict(float)
        self.returns = defaultdict(list)
        
    def learn_episode(self, episode_trajectory):
        """
        episode_trajectory: lista di tuple (state, reward)
        Implementa First-visit Monte Carlo prediction
        """
        G = 0
        
        # Scorre l'episodio al contrario
        for t in reversed(range(len(episode_trajectory))):
            state, reward = episode_trajectory[t]
            G = self.gamma * G + reward
            
            # Controlla first-visit (se lo stato non è apparso prima in questo episodio)
            states_before_t = [step[0] for step in episode_trajectory[:t]]
            if state not in states_before_t:
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
