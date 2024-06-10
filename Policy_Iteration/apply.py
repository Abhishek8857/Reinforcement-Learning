from env import Environment
from agent import Agent
from debug import evaluate

env = Environment(min_score_prob = 0.1, max_score_prob = 0.95, line_position = 2, field_length = 8, render_mode = "rgb_array")
test_agent = Agent(env, gamma = 0.99)
test_agent.train()

video_file_2 = "Policy_Iteration/basketball_training.mp4"
evaluate(env, test_agent.policy, video_file_2)
