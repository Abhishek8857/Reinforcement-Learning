import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.animation import FuncAnimation  
from render_util import visualize, plot_action_value

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SARSA_Agent:
    def __init__(self, env, learning_rate=0.05, gamma=1.0, epsilon=0.1):
        """ Initializes the environment and defines dynamics."""
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = 0.1
        
        self.states = env.observation_space.n 
        self.actions = env.action_space.n

        self.q = np.zeros((self.states, self.actions))
        
        
    def get_best_action(self, state, q):
        """ Return the best action based on the Q-function.
    
        Args: 
            obs: state of the environment
            q: The chosen Q-Function, a numpy array of shape (num_states, num_actions)
        Returns:
            best_action: Chosen action
        """
        best_action = np.random.choice(np.flatnonzero(np.isclose(q[state], (q[state]).max(), rtol=0.01))) 
        return best_action    
    
    
    def epsilon_greedy_policy(self, state, q):
        """ Return an action based on the Q-function and probability self.epsilon.
        
        The action should be random with probability self.epsilon, or otherwise the best action based on the Q-function.
        
        Args: 
            obs: state of the environment
            q: The chosen Q-Function, a numpy array of shape (num_states, num_actions)
        Returns:
            action: Chosen action
        """
        if np.random.random() > self.epsilon:
            greedy_action = np.random.choice(np.flatnonzero(np.isclose(q[state], (q[state]).max(), rtol=0.01)))
        else:
            greedy_action = np.random.choice(len(q[state]))
        return greedy_action
    
    
    def train(self, num_episodes):
        """ Trains the agent with the q algorithm.
        Args: 
        num_episodes: Number of episodes used until training stops"""
        for i in range(num_episodes + 1):
            state, info = self.env.reset()
            converged = False
            while not converged:
                action = self.epsilon_greedy_policy(state, self.q)
                next_state, reward, converged, truncated, info = self.env.step(action)
                next_action = self.epsilon_greedy_policy(next_state, self.q)
                
                val = reward + ((self.gamma * self.q[next_state][next_action]) - self.q[state][action])
                self.q[state][action] = self.q[state][action] + self.learning_rate * val
                
                state = next_state
                action = next_action
            
                
    def evaluate(self, env, num_runs=5):
        """ Evaluates the agent in the environment.

        Args:
            env: Environment we want to use. 
            file: File used for storing the video.
            num_runs: Number of runs displayed
        Returns:
            converged: Info about whether the last run is done.
            reward: The reward the agent gathered in the last step.
        """ 
        
        for i in range(num_runs):
            converged = False
            state, info = env.reset()
            rewards_collected = []

            while not converged:
                action = self.get_best_action(state, self.q)
                next_state, reward, converged, truncated, info = env.step(action)
                rewards_collected.append(reward)

                next_action = self.get_best_action(next_state, self.q)
                
                state = next_state
                action = next_action
                
            
        return rewards_collected
    
    
setattr(SARSA_Agent, 'visualize', visualize)
setattr(SARSA_Agent, 'plot_action_value', plot_action_value)
