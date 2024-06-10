import gymnasium as gym
from agent import Agent
import custom_envs

map = ["SFFH", "FFGH", "HFFH", "HFFF"]
test_env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
test_agent = Agent(test_env)

test_video = "Double_Q_Learning/test_run.mp4"
test_run = test_agent.evaluate(test_env, test_video)[0] 
