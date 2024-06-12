import gymnasium as gym
from agent import Agent
import custom_envs
import time 

# Set the number of training runs
training_runs = 10000

# Set up the environment and the Agent
map = ["SFFF", "FHFH", "FHFH", "FFFG"]
env = gym.make('CustomFrozenLake-v1', render_mode='rgb_array', desc=map, is_slippery=False) 
env.reset()
final_agent = Agent(env, gamma=0.9)
final_agent.visualize(0)

# Record the time and train the Agent
start_time = time.time()
final_agent.train(training_runs)
end_time = time.time()

# Time required for convergence
total_time = end_time - start_time
print(total_time)

# Visualise the agent and save the video file generated
final_agent.visualize(training_runs)
video = "Double_Q_Learning/final_run.mp4"
final_agent.evaluate(env, video, num_runs=1)

