import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from mc_prediction import MCPrediction
from td_prediction import TDPrediction
from utils import Discretizer, plot_learning_curves

def train_agent(agent_class, env, discretizer, num_episodes=500, seed=42, **kwargs):
    np.random.seed(seed)
    agent = agent_class(action_space=env.action_space.n, **kwargs)
    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = discretizer.discretize(obs)
        
        if isinstance(agent, SARSAAgent):
            action = agent.act(state)
        
        ep_reward = 0
        done = False
        
        while not done:
            if isinstance(agent, QLearningAgent):
                action = agent.act(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = discretizer.discretize(next_obs)
                agent.learn(state, action, reward, next_state, done)
            
            elif isinstance(agent, SARSAAgent):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = discretizer.discretize(next_obs)
                next_action = agent.act(next_state)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action
            
            state = next_state
            ep_reward += reward
            
        agent.decay_epsilon()
        rewards.append(ep_reward)
        
    return rewards, agent

def run_control_experiments():
    env = gym.make('CartPole-v1')
    discretizer = Discretizer()
    seeds = [42, 123, 456]
    num_episodes = 500
    
    results = {}
    
    # Q-Learning
    q_rewards = []
    for seed in seeds:
        print(f"Training Q-Learning Seed {seed}...")
        rews, _ = train_agent(QLearningAgent, env, discretizer, num_episodes, seed)
        q_rewards.append(rews)
    results['Q-Learning'] = q_rewards
    
    # SARSA
    sarsa_rewards = []
    for seed in seeds:
        print(f"Training SARSA Seed {seed}...")
        rews, _ = train_agent(SARSAAgent, env, discretizer, num_episodes, seed)
        sarsa_rewards.append(rews)
    results['SARSA'] = sarsa_rewards
    
    plot_learning_curves(results, 'Q-Learning vs SARSA on CartPole', 'q_vs_sarsa.png')

def run_sensitivity_analysis():
    env = gym.make('CartPole-v1')
    discretizer = Discretizer()
    seeds = [42, 123, 456]
    num_episodes = 300
    
    # Alpha sensitivity
    alphas = [0.05, 0.1, 0.2]
    alpha_results = {}
    for alpha in alphas:
        rews_all = []
        for seed in seeds:
            rews, _ = train_agent(QLearningAgent, env, discretizer, num_episodes, seed, alpha=alpha)
            rews_all.append(rews)
        alpha_results[f'Alpha={alpha}'] = rews_all
    plot_learning_curves(alpha_results, 'Q-Learning Sensitivity to Learning Rate (alpha)', 'sensitivity_alpha.png')
    
    # Gamma sensitivity
    gammas = [0.9, 0.99, 1.0]
    gamma_results = {}
    for gamma in gammas:
        rews_all = []
        for seed in seeds:
            rews, _ = train_agent(QLearningAgent, env, discretizer, num_episodes, seed, gamma=gamma)
            rews_all.append(rews)
        gamma_results[f'Gamma={gamma}'] = rews_all
    plot_learning_curves(gamma_results, 'Q-Learning Sensitivity to Discount Factor (gamma)', 'sensitivity_gamma.png')

    # Epsilon decay sensitivity
    decays = [0.9, 0.95, 0.995]
    decay_results = {}
    for decay in decays:
        rews_all = []
        for seed in seeds:
            rews, _ = train_agent(QLearningAgent, env, discretizer, num_episodes, seed, epsilon_decay=decay)
            rews_all.append(rews)
        decay_results[f'Epsilon Decay={decay}'] = rews_all
    plot_learning_curves(decay_results, 'Q-Learning Sensitivity to Epsilon Decay', 'sensitivity_epsilon.png')

def evaluate_prediction():
    env = gym.make('CartPole-v1')
    discretizer = Discretizer()
    
    # Usiamo una policy euristica fissa per poter valutare i valori
    # Se il palo pende a destra, vai a destra, altrimenti sinistra
    def policy(obs):
        return 1 if obs[2] > 0 else 0
        
    mc_agent = MCPrediction()
    td_agent = TDPrediction(alpha=0.05)
    
    # Fissiamo uno stato centrale comune per monitorare la stima del valore nel tempo
    # Pos = 0, Vel = 0, Angle = 0, AngVel = 0
    central_state = discretizer.discretize([0.0, 0.0, 0.0, 0.0])
    
    mc_v_history = []
    td_v_history = []
    
    num_episodes = 500
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        state = discretizer.discretize(obs)
        done = False
        trajectory = []
        
        while not done:
            action = policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretizer.discretize(next_obs)
            
            trajectory.append((state, reward))
            
            # TD(0) step
            td_agent.learn(state, reward, next_state, done)
            
            state = next_state
            obs = next_obs
            
        # MC step (a fine episodio)
        mc_agent.learn_episode(trajectory)
        
        mc_v_history.append(mc_agent.V[central_state])
        td_v_history.append(td_agent.V[central_state])
        
    plt.figure(figsize=(10, 6))
    plt.plot(mc_v_history, label='Monte Carlo V(s0)', alpha=0.7)
    plt.plot(td_v_history, label='TD(0) V(s0)', alpha=0.7)
    plt.title('MC vs TD(0) Prediction - Central State Value over time')
    plt.xlabel('Episodes')
    plt.ylabel('V(s0)')
    plt.legend()
    plt.grid()
    plt.savefig('mc_vs_td.png')
    plt.close()

if __name__ == "__main__":
    print("Running Control Experiments (Q-Learning vs SARSA)...")
    run_control_experiments()
    
    print("\nRunning Sensitivity Analysis...")
    run_sensitivity_analysis()
    
    print("\nRunning Prediction Experiments (MC vs TD(0))...")
    evaluate_prediction()
    
    print("\nAll experiments finished. Check the generated PNG plots in the current directory.")
