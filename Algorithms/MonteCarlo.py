import gymnasium as gym
import numpy as np

from Utils.policy import discretize, build_v_table, heuristic_policy


## Monte Carlo Prediction per stimare una value function V
def montecarlo_prediction(env_name: str, bins_list: list, num_episodes: int = 5000,
                          render: bool = False, gamma: float = 0.99,
                          alpha: float = 0.1, use_first_visit: bool = True):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    n_actions = env.action_space.n
    actions = list(range(n_actions))
    v_table = build_v_table(bins_list)
    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()

        ## Monte Carlo aggiorna solo a fine episodio quindi dobbiamo tenere traccia
        ## degli stati e dei reward
        episode_states = []
        episode_rewards = []
        done = False

        while not done:
            state = discretize(obs, bins_list)
            episode_states.append(state)

            ## Usiamo la policy che sappiamo non cambiera' essendo Prediction
            action = heuristic_policy(env_name, obs, actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)

        ## Anche nei colab di esercitazione avevamo il first visit quindi lo inseriamo
        if use_first_visit:
            visited = set()
            for t, s in enumerate(episode_states):
                if s in visited:
                    continue
                visited.add(s)

                # Calcola G_t = somma scontata dei reward da t in poi
                G = 0.0
                for k in range(len(episode_rewards) - 1, t - 1, -1):
                    G = gamma * G + episode_rewards[k]

                v_table[s] += alpha * (G - v_table[s])
        else:
            G = 0.0
            for s, r in zip(reversed(episode_states), reversed(episode_rewards)):
                G = gamma * G + r
                v_table[s] += alpha * (G - v_table[s])

        episode_reward = sum(episode_rewards)
        reward_history.append(episode_reward)

        ## Per debugging: Stampa progresso ogni 100 episodi
        # if (episode + 1) % 100 == 0:
        #     avg_reward = np.mean(reward_history[-100:])
        #     print(f"Episodio {episode + 1}/{num_episodes} | "
        #           f"Reward medio (ultimi 100): {avg_reward:.1f}")

    env.close()
    # print(f"\nMonte Carlo prediction completata! Reward medio finale (ultimi 100): "
    #       f"{np.mean(reward_history[-100:]):.1f}")
    return v_table, reward_history
