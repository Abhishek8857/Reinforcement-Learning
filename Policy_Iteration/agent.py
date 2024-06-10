import numpy as np
from env import Environment
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env, gamma=0.9, update_threshold=1e-6):
        """ Initializes the Agent.
        
        The agent takes properties of the environment and stores them for training.

        Args:
            env: Environment used for training.
            gamma: Discount factor.
            update_threshold: Stopping distance for updates of the value function.
        """
        self.mdp = (env.unwrapped.mdp, env.observation_space.n, env.action_space.n)
        self.update_threshold = update_threshold
        self.state_value_fn = np.zeros(self.mdp[1])
        
        self.policy = []
        
        for state in range(self.mdp[1]):
            random_entry = np.random.randint(0, 1)
            self.policy.append([0 for i in range(self.mdp[2])])
            self.policy[state][random_entry] = 1

        self.gamma = gamma
        self.iteration = 0
        
        
    def reset(self):
        """Resets the agent"""
        self.state_value_fn = np.zeros(self.mdp[1])
        self.policy = []
        for state in range(self.mdp[1]):
            random_entry = np.random.randint(0, 1)
            self.policy.append([0 for _ in range(self.mdp[2])])
            self.policy[state][random_entry] = 1
        self.iteration = 0
    
    
    def get_greedy_action(self, state):
        """ Choose an action based on the policy. """
        max_value = max(self.policy[state])
        best_actions_list = []
        for index, action in enumerate(self.policy[state]):
            if action == max_value:
                best_actions_list.append(index)
        return np.random.choice(best_actions_list)
    
    
    def visualize(self):
        """ Visualize the Q-function. """
        x_axis = 1
        y_axis = self.mdp[1]-2 
        vmin = min(self.state_value_fn)
        vmax = max(self.state_value_fn)
        X1 = np.reshape(self.state_value_fn[:-2], (x_axis, y_axis))
        fig, ax = plt.subplots(1, 1)
        cmap = plt.colormaps["Blues_r"]
        cmap.set_under("black")
        img = ax.imshow(X1, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('off')
        ax.set_title("Values of the state value function on the field")
        for i in range(x_axis):
            for j in range(y_axis):
                ax.text(j, i, str(X1[i][j])[:4], fontsize=12, color='black', ha='center', va='center')
        plt.show()
   
        
    def render_policy(self):
        """ Print the current policy. """
        print('Policy of the agent:')
        out = ' | '
        render = out
        for i in range(self.mdp[1]-2):
            token = ""
            if self.policy[i][0] > 0:   # move
                token += "Move"
            if self.policy[i][1] > 0:   # up
                token += "Throw"
            if len(token) > 5:
                token = 'Move or Throw'
            render += token + out
        print(render) 

    
    def train(self):
        converged = False
        sweeps = 0
        while not converged:
            sweeps += self.policy_evaluation()
            converged = self.policy_improvement()
            self.iteration += 1
        print("Sweeps required for convergence: ", sweeps)
        print("Number of iterations required for convergence: ", self.iteration)
    
    
    def policy_evaluation(self):
        converged = False
        sweeps = 0
        
        while not converged:
            delta = 0
            sweeps += 1
            
            for state in range(self.mdp[1]):
                old_state_val = self.state_value_fn[state]
                new_state_val = 0
                
                for action in range(self.mdp[2]):
                    new_state_val += self.get_policy_value(state, action)
                    
                self.state_value_fn[state] = new_state_val
                delta = max(delta, np.abs(old_state_val - self.state_value_fn[state]))
            
            if delta < self.update_threshold:
                converged = True
                break
            
        return sweeps
            
                        
    def get_action_value(self, state, action):
        action_val = 0
        
        for transitions in self.mdp[0][state][action]:
            transition_prob = transitions[0]
            next_state = transitions[1]
            reward = transitions[2]
            action_val += transition_prob * (reward + self.gamma * self.state_value_fn[next_state])
        
        return action_val
    
    
    def get_policy_value(self, state, action):
        policy_val = 0
        
        for transitions in self.mdp[0][state][action]:
            transition_prob = transitions[0]
            next_state = transitions[1]
            reward = transitions[2]
            policy_val += self.policy[state][action] * transition_prob * (reward + self.gamma * self.state_value_fn[next_state])
        
        return policy_val
    
    
    def policy_improvement(self):
        converged = True
        current_policy = self.policy
        best_policy = []
        
        for state in range(self.mdp[1]):
            best_policy.append([0 for i in range(self.mdp[2])])
            
            action_values = []
            for action in range(self.mdp[2]):
                action_values.append(self.get_action_value(state, action))
                
            best_actions = []
            for index, action in enumerate(action_values):
                if action == max(action_values):
                    best_actions.append(index)
                    
            for index in best_actions:
                best_policy[state][index] = 1
                 
            best_policy[state] = [best_policy[state][action] / len(best_actions)
                                  for action in range(self.mdp[2])]

            if not np.array_equal(current_policy[state], best_policy[state]):
                converged = False
            
            self.policy[state] = best_policy[state]
        return converged
   
   

