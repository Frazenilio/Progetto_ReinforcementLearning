import gymnasium as gym
import numpy as np

from Utils.policy import discretize, build_v_table, heuristic_policy


## Uso di TD per Prediction
def temporal_difference(env_name: str, bins_list: list, num_episodes: int = 5000,
                        render: bool = False, gamma: float = 0.99,
                        alpha: float = 0.1):
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
        done = False
        state = discretize(obs, bins_list)
        episode_reward = 0.0

        while not done:
            ## TD non impara policy quindi ne usiamo una data
            action = heuristic_policy(env_name, obs, actions)

            ## Eseguiamo l'azione
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins_list)

            ## Update del V(s) visto che TD(0) aggiorna subito dopo aver eseguito il passo
            v_table[state] = v_table[state] + alpha * (
                reward + gamma * v_table[next_state] - v_table[state]
            )
            state = next_state
            episode_reward += reward

        reward_history.append(episode_reward)

        # Debugging: Stampa progresso ogni 100 episodi
        # if (episode + 1) % 100 == 0:
        #     avg_reward = np.mean(reward_history[-100:])
        #     print(f"Episodio {episode + 1}/{num_episodes} | "
        #           f"Reward medio (ultimi 100): {avg_reward:.1f}")

    env.close()
    # print(f"\nTD(0) prediction completata! Reward medio finale (ultimi 100): "
    #       f"{np.mean(reward_history[-100:]):.1f}")
    return v_table, reward_history
