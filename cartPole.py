import gymnasium as gym
import ale_py
import numpy as np
import itertools
#import matplotlib.pyplot as plt
MAX_VELOCITY = 3.5
MAX_ANGULAR_VELOCITY = 3.5
MAX_POSITION = 2.4
MAX_ANGLE = 0.21
MIN_VELOCITY = -3.5
MIN_ANGULAR_VELOCITY = -3.5
MIN_POSITION = -2.4
MIN_ANGLE = -0.21


def discretize(obs, positions, angles, velocities, angular_velocities):
    """Mappa un'osservazione continua ai valori discreti più vicini (bin).
    
    Per ogni componente dell'osservazione [pos, vel, ang, ang_vel],
    trova il valore più vicino nell'array di bin corrispondente.
    """
    pos = positions[np.argmin(np.abs(positions - obs[0]))]
    vel = velocities[np.argmin(np.abs(velocities - obs[1]))]
    ang = angles[np.argmin(np.abs(angles - obs[2]))]
    ang_vel = angular_velocities[np.argmin(np.abs(angular_velocities - obs[3]))]
    return (pos, ang, vel, ang_vel)


def q_learning(env_name: str, num_episodes: int = 5000, render: bool = False,
               num_positions: int = 24, num_angles: int = 30,
               num_velocities: int = 24, num_angular_velocities: int = 24,
               alpha: float = 0.2, gamma: float = 0.99,
               epsilon: float = 1.0, epsilon_min: float = 0.01,
               epsilon_decay: float = 0.999):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Crea i bin per discretizzare lo spazio degli stati continuo
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    angles = np.linspace(MIN_ANGLE, MAX_ANGLE, num_angles)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    angular_velocities = np.linspace(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY, num_angular_velocities)
    actions = [0, 1]

    # Q-table: dizionario (pos, ang, vel, ang_vel, action) -> 0.0
    q_table = {
        (p, a, v, av, act): 0.0
        for p, a, v, av, act in itertools.product(positions, angles, velocities, angular_velocities, actions)
    }

    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        # Discretizza l'osservazione iniziale
        state = discretize(obs, positions, angles, velocities, angular_velocities)

        episode_reward = 0.0
        done = False

        while not done:
            # 1) Epsilon-greedy: esplora con probabilità epsilon, altrimenti sfrutta
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # esplorazione
            else:
                # Sfruttamento: scegli l'azione con Q-value massimo
                q_values = [q_table[(*state, a)] for a in actions]
                action = actions[np.argmax(q_values)]

            # DEBUG: stampa osservazione e azione ad ogni passo
            #action_name = "← Sinistra" if action == 0 else "→ Destra"
            #print(f"  Pos: {obs[0]:+.4f} | Vel: {obs[1]:+.4f} | "
            #      f"Angolo: {np.degrees(obs[2]):+.2f}° | Vel.Ang: {np.degrees(obs[3]):+.2f}°/s | "
            #      f"Azione: {action_name}")

            # 2) Esegui l'azione nell'ambiente
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, positions, angles, velocities, angular_velocities)

            # 3) Q-learning update: Q(s,a) += α * [r + γ * max_a' Q(s',a') - Q(s,a)]
            # Alpha adattivo: decresce col progredire del training per stabilità
            adaptive_alpha = max(0.01, alpha * (1 - episode / num_episodes))
            best_next_q = max(q_table[(*next_state, a)] for a in actions)
            td_target = reward + gamma * best_next_q * (1 - int(done))
            td_error = td_target - q_table[(*state, action)]
            q_table[(*state, action)] += adaptive_alpha * td_error

            obs = next_obs
            state = next_state
            episode_reward += reward

        # Stampa motivo fine episodio e reward
        if terminated:
            if abs(obs[2]) > np.radians(12):
                reason = "Palo caduto (angolo > 12°)"
            elif abs(obs[0]) > 2.4:
                reason = "Carrello fuori dai limiti (|pos| > 2.4)"
            else:
                reason = "Terminato"
        else:
            reason = "Troncato (raggiunto limite step)"
        print(f"Episodio {episode + 1} | Reward: {episode_reward:.0f} | Fine: {reason}")
        # Decay epsilon dopo ogni episodio (esplora meno col tempo)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        # Stampa progresso ogni 100 episodi
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episodio {episode + 1}/{num_episodes} | "
                  f"Reward medio (ultimi 100): {avg_reward:.1f} | "
                  f"Epsilon: {epsilon:.3f}")

    env.close()
    print(f"\nTraining completato! Reward medio finale (ultimi 100): "
          f"{np.mean(reward_history[-100:]):.1f}")
    return q_table, reward_history

def sarsa(env_name: str, num_episodes: int = 5000, render: bool = False,
               num_positions: int = 24, num_angles: int = 30,
               num_velocities: int = 24, num_angular_velocities: int = 24,
               alpha: float = 0.2, gamma: float = 0.99,
               epsilon: float = 1.0, epsilon_min: float = 0.01,
               epsilon_decay: float = 0.999):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Crea i bin per discretizzare lo spazio degli stati continuo
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    angles = np.linspace(MIN_ANGLE, MAX_ANGLE, num_angles)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    angular_velocities = np.linspace(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY, num_angular_velocities)
    actions = [0, 1]

    # Q-table: dizionario (pos, ang, vel, ang_vel, action) -> 0.0
    q_table = {
        (p, a, v, av, act): 0.0
        for p, a, v, av, act in itertools.product(positions, angles, velocities, angular_velocities, actions)
    }

    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        # Discretizza l'osservazione iniziale
        state = discretize(obs, positions, angles, velocities, angular_velocities)

        episode_reward = 0.0
        done = False
        
        #Q(S,a)
        # 1) Epsilon-greedy: esplora con probabilità epsilon, altrimenti sfrutta
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # esplorazione
        else:
            # Sfruttamento: scegli l'azione con Q-value massimo
            q_values = [q_table[(*state, a)] for a in actions]
            action = actions[np.argmax(q_values)]
        
        while not done:
            
            # DEBUG: stampa osservazione e azione ad ogni passo
            #action_name = "← Sinistra" if action == 0 else "→ Destra"
            #print(f"  Pos: {obs[0]:+.4f} | Vel: {obs[1]:+.4f} | "
            #      f"Angolo: {np.degrees(obs[2]):+.2f}° | Vel.Ang: {np.degrees(obs[3]):+.2f}°/s | "
            #      f"Azione: {action_name}")

            # 2) Esegui l'azione nell'ambiente
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize(obs, positions, angles, velocities, angular_velocities)

            #Q(S',a')
             # 1) Epsilon-greedy: esplora con probabilità epsilon, altrimenti sfrutta
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()  # esplorazione
            else:
                # Sfruttamento: scegli l'azione con Q-value massimo
                q_values = [q_table[(*next_state, a)] for a in actions]
                next_action = actions[np.argmax(q_values)]

            # DEBUG: stampa osservazione e azione ad ogni passo
            #action_name = "← Sinistra" if action == 0 else "→ Destra"
            #print(f"  Pos: {obs[0]:+.4f} | Vel: {obs[1]:+.4f} | "
            #      f"Angolo: {np.degrees(obs[2]):+.2f}° | Vel.Ang: {np.degrees(obs[3]):+.2f}°/s | "
            #      f"Azione: {action_name}")


            # 3) SARSA update: Q(s,a) += α * [r + γ * Q(s',a') - Q(s,a)]
            # Alpha adattivo: decresce col progredire del training per stabilità
            adaptive_alpha = max(0.01, alpha * (1 - episode / num_episodes))
           
            next_q_value = q_table[(*next_state, next_action)]
            td_target = reward + gamma * next_q_value * (1 - int(done))
            td_error = td_target - q_table[(*state, action)]
            q_table[(*state, action)] += adaptive_alpha * td_error

            state = next_state
            action = next_action
            episode_reward += reward

        # Stampa motivo fine episodio e reward
        if terminated:
            if abs(obs[2]) > np.radians(12):
                reason = "Palo caduto (angolo > 12°)"
            elif abs(obs[0]) > 2.4:
                reason = "Carrello fuori dai limiti (|pos| > 2.4)"
            else:
                reason = "Terminato"
        else:
            reason = "Troncato (raggiunto limite step)"
        print(f"Episodio {episode + 1} | Reward: {episode_reward:.0f} | Fine: {reason}")
        # Decay epsilon dopo ogni episodio (esplora meno col tempo)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        # Stampa progresso ogni 100 episodi
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episodio {episode + 1}/{num_episodes} | "
                  f"Reward medio (ultimi 100): {avg_reward:.1f} | "
                  f"Epsilon: {epsilon:.3f}")

    env.close()
    print(f"\nTraining completato! Reward medio finale (ultimi 100): "
          f"{np.mean(reward_history[-100:]):.1f}")
    return q_table, reward_history

def montecarlo_prediction(env_name: str, num_episodes: int = 5000, render: bool = False,
               num_positions: int = 24, num_angles: int = 30,
               num_velocities: int = 24, num_angular_velocities: int = 24,
               gamma: float = 0.99,alpha:float=0.1):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Crea i bin per discretizzare lo spazio degli stati continuo
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    angles = np.linspace(MIN_ANGLE, MAX_ANGLE, num_angles)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    angular_velocities = np.linspace(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY, num_angular_velocities)

    # V-table: dizionario stato -> valore stimato
    v_table = {
        (p, a, v, av): 0.0
        for p, a, v, av in itertools.product(positions, angles, velocities, angular_velocities)
    }

    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()

        # === FASE 1: Genera l'episodio completo ===
        # Raccogli tutta la traiettoria (stati e reward) PRIMA di fare update
        episode_states = []
        episode_rewards = []
        done = False

        while not done:
            state = discretize(obs, positions, angles, velocities, angular_velocities)
            episode_states.append(state)

            # Policy random (prediction = valutiamo una policy fissa)
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)

        G = 0.0
        for s, r in zip(reversed(episode_states), reversed(episode_rewards)):
            G = gamma * G + r
            v_table[s] += alpha * (G - v_table[s])

        episode_reward = sum(episode_rewards)
        reward_history.append(episode_reward)

        # Stampa motivo fine episodio e reward
        if terminated:
            if abs(obs[2]) > np.radians(12):
                reason = "Palo caduto (angolo > 12°)"
            elif abs(obs[0]) > 2.4:
                reason = "Carrello fuori dai limiti (|pos| > 2.4)"
            else:
                reason = "Terminato"
        else:
            reason = "Troncato (raggiunto limite step)"
        print(f"Episodio {episode + 1} | Reward: {episode_reward:.0f} | Fine: {reason}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episodio {episode + 1}/{num_episodes} | "
                  f"Reward medio (ultimi 100): {avg_reward:.1f}")

    env.close()
    print(f"\nPrediction completata! Reward medio finale (ultimi 100): "
          f"{np.mean(reward_history[-100:]):.1f}")
    return v_table

def temporal_difference(env_name: str, num_episodes: int = 5000, render: bool = False,
               num_positions: int = 24, num_angles: int = 30,
               num_velocities: int = 24, num_angular_velocities: int = 24,
               gamma: float = 0.99,alpha:float=0.1):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    # Crea i bin per discretizzare lo spazio degli stati continuo
    positions = np.linspace(MIN_POSITION, MAX_POSITION, num_positions)
    angles = np.linspace(MIN_ANGLE, MAX_ANGLE, num_angles)
    velocities = np.linspace(MIN_VELOCITY, MAX_VELOCITY, num_velocities)
    angular_velocities = np.linspace(MIN_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY, num_angular_velocities)

    # V-table: dizionario stato -> valore stimato
    v_table = {
        (p, a, v, av): 0.0
        for p, a, v, av in itertools.product(positions, angles, velocities, angular_velocities)
    }

    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        # === FASE 1: Genera l'episodio completo ===
        # Raccogli tutta la traiettoria (stati e reward) PRIMA di fare update
        done = False
        state = discretize(obs, positions, angles, velocities, angular_velocities)
        while not done:
           
            # Policy random (prediction = valutiamo una policy fissa)
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, positions, angles, velocities, angular_velocities)
            v_table[state] = v_table[state] + alpha * (reward + gamma * v_table[next_state] - v_table[state])
            state = next_state
        """
        # Stampa motivo fine episodio e reward
        if terminated:
            if abs(obs[2]) > np.radians(12):
                reason = "Palo caduto (angolo > 12°)"
            elif abs(obs[0]) > 2.4:
                reason = "Carrello fuori dai limiti (|pos| > 2.4)"
            else:
                reason = "Terminato"
        else:
            reason = "Troncato (raggiunto limite step)"
        print(f"Episodio {episode + 1} | Reward: {episode_reward:.0f} | Fine: {reason}")

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episodio {episode + 1}/{num_episodes} | "
                  f"Reward medio (ultimi 100): {avg_reward:.1f}")"""

    env.close()
    # print(f"\nPrediction completata! Reward medio finale (ultimi 100): "
    # f"{np.mean(reward_history[-100:]):.1f}")
    return v_table


if __name__ == "__main__":
    # Training Q-learning su CartPole con discretizzazione
    v_table = temporal_difference("CartPole-v1", num_episodes=5000, render=False)