import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from DQ_Learning_agent import DQ_Agent
from Q_Learning_agent import Q_Agent
from SARSA_agent import SARSA_Agent
import custom_envs
import time 


def get_data(agent, runs, env):
    start_time = time.time()
    agent.train(runs)
    rewards = agent.evaluate(env)
    end_time = time.time()
    
    return end_time - start_time, rewards


def run():
    # Set the number of training runs
    training_runs = 10000

    # Set up the environment and the Agent
    map = ["SFFF", "FHFH", "FHFH", "FFFG"]
    env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
    env.reset()
    q_learning = Q_Agent(env, gamma=0.9)
    sarsa_learning = SARSA_Agent(env, gamma=0.9)
    dq_learning = DQ_Agent(env, gamma=0.9)
    # Time required for convergence

    q_time, q_rewards = get_data(q_learning, training_runs, env)
    sarsa_time, sarsa_rewards = get_data(sarsa_learning, training_runs, env)
    dq_time, dq_rewards = get_data(dq_learning, training_runs, env)

    # Visualise the agent and save the video file generated
    # final_agent.visualize(training_runs)
    # video = "Double_Q_Learning/final_run.mp4"
    # final_agent.evaluate(env, video, num_runs=1)
    
    return [q_time, sarsa_time, dq_time], [q_rewards, sarsa_rewards, dq_rewards]


if __name__ == "__main__":
    times, rewards = run()

    # Plot the times on a graph
    agent_names = ['Q-Learning', 'SARSA', 'Double Q-Learning']

    plt.figure(figsize=(8, 5))
    plt.barh(agent_names, times, height=0.3, color=['blue', 'green', 'red'])
    plt.xlabel('Agent')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken by Each Agent for Convergence')
    
    for i in rewards:
        print(i)
    plt.figure(figsize=(10, 5))
    for i, agent_name in enumerate(agent_names):
        plt.plot(rewards[i], label=agent_name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.show()
    