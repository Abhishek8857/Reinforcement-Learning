import numpy as np
import warnings
import matplotlib.pyplot as plt
from render_util import visualize, plot_action_value
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DQ_Agent:
    def __init__(self, env, gamma=1.0, learning_rate=0.1, epsilon=0.1):
        """ Initializes the environment and defines dynamics."""
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.states = self.env.observation_space.n
        self.actions = self.env.action_space.n 
        
        self.q_1 = np.zeros(shape=(self.states, self.actions))
        self.q_2 = np.zeros(shape=(self.states, self.actions)) 
        
        
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
    
    
    def epsilon_greedy_action(self, state, q):
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
        """ Trains the agent with the double-q algorithm.
        Args: 
        num_episodes: Number of episodes used until training stops"""
        
        for i in range(num_episodes + 1):
            state, info = self.env.reset()
            converged = False
            
            while not converged:
                action = int(self.epsilon_greedy_action(state, q=self.q_1 + self.q_2))
                next_state, reward, converged, truncated, info = self.env.step(action)
                if np.random.random() > 0.5:
                    val = self.q_2[next_state][np.argmax(self.q_1[next_state]).astype(int)]
                    self.q_1[state][action] = self.q_1[state][action] + self.learning_rate * (reward + (self.gamma * val) - self.q_1[state][action])
                else:
                    val = self.q_1[next_state][np.argmax(self.q_2[next_state]).astype(int)]
                    self.q_2[state][action] = self.q_2[state][action] + self.learning_rate * (reward + (self.gamma * val) - self.q_2[state][action])

                state = next_state


    def evaluate(self, env, num_runs=5):
        """ Evaluates the agent in the environment.

        Args:
            env: Environment we want to use. 
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
                combined_q = self.q_1 + self.q_2
                action = self.get_best_action(state, combined_q)
                
                next_state, reward, converged, truncated, info = env.step(action)
                rewards_collected.append(reward)

                state = next_state

        return rewards_collected
    
    

setattr(DQ_Agent, 'visualize', visualize)
setattr(DQ_Agent, 'plot_action_value', plot_action_value)
