import gymnasium as gym
import numpy as np

from Utils.policy import discretize, build_q_table, epsilon_greedy_action


## SARSA per apprendimento di una policy
def sarsa(env_name: str, bins_list: list, num_episodes: int = 5000,
          render: bool = False, alpha: float = 0.2, gamma: float = 0.99,
          epsilon: float = 0.1):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    n_actions = env.action_space.n
    q_table = build_q_table(bins_list, n_actions)
    reward_history = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        state = discretize(obs, bins_list)

        episode_reward = 0.0
        done = False

        ## SARSA loopa ricordando la next_action e la usa al prossimo timestep quindi all'inizio entriamo
        ## con gia' l'azione scelta
        action = epsilon_greedy_action(q_table, state, epsilon, n_actions)

        while not done:
            ## Eseguiamo l'azione
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretize(obs, bins_list)

            ## Scegliamo subito l'azione PRIMA di scegliere la prossima azione
            next_action = epsilon_greedy_action(q_table, next_state, epsilon, n_actions)

            ## Prendiamo il prossimo q (q(s',a'), a' e' la prossima azione)
            next_q_value = q_table[next_state][next_action]
            ## USiamo la formula
            td_target = reward + gamma * next_q_value * (1 - int(done))
            td_error = td_target - q_table[state][action]
            ## Ed incrementiamo il q corrente
            q_table[state][action] += alpha * td_error

            state = next_state
            action = next_action
            episode_reward += reward

        reward_history.append(episode_reward)

        # Debugging: Stampa progresso ogni 100 episodi
        # if (episode + 1) % 100 == 0:
        #     avg_reward = np.mean(reward_history[-100:])
        #     print(f"Episodio {episode + 1}/{num_episodes} | "
        #           f"Reward medio (ultimi 100): {avg_reward:.1f} | "
        #           f"Epsilon: {epsilon:.3f}")

    env.close()
    # print(f"\nSARSA completato! Reward medio finale (ultimi 100): "
    #       f"{np.mean(reward_history[-100:]):.1f}")
    return q_table, reward_history
