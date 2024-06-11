import gymnasium as gym
from agent import Agent
import custom_envs

map = ["SFFH", "FFFH", "HFFH", "HFFG"]
env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
env.reset()
agent = Agent(env, gamma=0.9)
agent.train(num_episodes=5000)
agent.plot_action_value()
video = "Off_Policy_MC_with_WIS/final_run.mp4"
agent.evaluate(env, video, num_runs=5)
