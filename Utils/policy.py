import numpy as np
import itertools
from collections import defaultdict


## Discretizza un'osservazione facendola ricadere nel range opportuno
def discretize(obs, bins_list):
    discretized = []
    for value, bins in zip(obs, bins_list):
        idx = np.argmin(np.abs(bins - value))
        discretized.append(bins[idx])
    return tuple(discretized)

## Costruisce una q-table
def build_q_table(bins_list, n_actions):
    q_table = defaultdict(lambda: np.zeros(n_actions))
    ## Così forziamo il dict a creare quella chiave
    for state in itertools.product(*bins_list):
        _ = q_table[state]
    return q_table


## Costruisce una v-table
def build_v_table(bins_list):
    v_table = defaultdict(float)
    for state in itertools.product(*bins_list):
        _ = v_table[state]
    return v_table

## Utilizzo di una random policy: ritorna un'azione casuale tra quelle possibili
def random_policy(actions):
    return np.random.choice(actions)


## Utilizzo di un'azione greedy: ritorna l'indice dell'azione migliore (o una a caso tra le migliori se
## ce n'e' piu' di una)
def greedy_action(Q, state):
    qvals = Q[state]
    max_q = np.max(qvals)
    candidates = np.flatnonzero(np.isclose(qvals, max_q))
    return int(np.random.choice(candidates))


## Utilizza una policy epsilon-greedy: con probabilita' epsilon sceglie casualmente, altrimenti
## usa la greedy policy
def epsilon_greedy_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return greedy_action(Q, state)


def heuristic_policy(env_name, obs, actions):
    ## Cart Pole: quando il bastone e' inclinato verso destra, spingilo a sinistra.
    ## Viceversa se il bastone e' inclinato verso sinistra, spingilo a destra
    if "CartPole" in env_name:
        ## obs = [position, velocity, pole_angle, pole_velocity]
        angle = obs[2]
        if angle > 0:
            return 0  # Push left
        else:
            return 1  # Push right

    ## Mountain Car: quando la velocita' e' negativa, sta andando verso sinistra quindi accelera verso destra
    ## Viceversa, quando la velocita' e' positiva, accelera verso sinistra  
    elif "MountainCar" in env_name:
        ## obs = [position, velocity]
        velocity = obs[1]
        if velocity < 0:
            return 2  # Push right (opposto alla velocità negativa)
        elif velocity > 0:
            return 0  # Push left (opposto alla velocità positiva)
        else:
            return np.random.choice(actions)
    ## In caso di velocita' nulla, non potendo sapere se e' fermo o se ha raggiunto il "culmine"
    ## di una delle due salite, lo facciamo agire casualmente     
    else:
        return random_policy(actions)
