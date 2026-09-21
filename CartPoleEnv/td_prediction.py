import numpy as np
from collections import defaultdict

class TDPrediction:
    def __init__(self, alpha=0.1, gamma=0.99):
        self.alpha = alpha
        self.gamma = gamma
        self.V = defaultdict(float)
        
    def learn(self, state, reward, next_state, done):
        td_target = reward + self.gamma * self.V[next_state] * (1 - done)
        td_error = td_target - self.V[state]
        self.V[state] += self.alpha * td_error
