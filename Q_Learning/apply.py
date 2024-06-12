import gymnasium as gym
from agent import Agent
import custom_envs

training_runs = 10000
map = ["SFFF", "FHFH", "FHFH", "FFFG"]
env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
env.reset()
final_agent = Agent(env, gamma=0.9)

final_agent.visualize(0)
final_agent.train(training_runs)
final_agent.visualize(training_runs)
video = "Q_Learning/final_run.mp4"
final_agent.evaluate(env, video, num_runs=1)
