import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
import gymnasium as gym
from render_utils import render


class Environment(Env):
    def __init__(self, min_score_prob=0.0, max_score_prob=0.9,
                 line_position=3, field_length=10, render_mode=None):
        
        super(Environment, self).__init__()

        ############ Render #############
        self.render_mode = render_mode
        self.render_time = 1 # one image per second
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization. "
            )
        if self.render_mode != "text" and self.render_mode != None:
            self.render_width = 130
            self.window_size = ((field_length+1)*self.render_width, self.render_width)
            self.cell_size = (self.render_width, self.render_width)
            self.window_surface = None
        ############ Render #############
            
        self.state = 0
        self.laststate= None
        self.field_length = field_length
        self.line_position = line_position
        self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.observation_space = spaces.Discrete(self.field_length + 2)
        self.mdp = {}
        
        # Define loop variables
        self.terminal_state = self.field_length + 2
        self.temp_min = min_score_prob
        self.temp_max = max_score_prob

        # Loop 1
        for i in range(self.terminal_state):
            self.mdp[i] = {} # Create a dictionary
            # Loop 2
            for j in range(2):
                self.mdp[i][j] = [] # Create Actions

            self.reward = 3 if line_position > i else 2 # Select reward based on line position
            self.state = False if (i < self.terminal_state - 3) else True # Select state based on terminal state

            # Add conditions for Action 0
            if i == (field_length - 1):
                self.mdp[i][0].append((1, self.field_length + 1, 0, self.state))
            elif i >= field_length:
                self.mdp[i][0].append((1, i, 0, self.state))
            else:
                self.mdp[i][0].append((1, i + 1, 0, self.state))

            # Add conditions for Action 1
            if i < self.field_length:
                self.mdp[i][1].append((round(self.temp_min, 5), self.field_length, self.reward, True))                         
                self.mdp[i][1].append((round(1 - self.temp_min, 5), self.field_length + 1, 0, True))                            
                self.temp_min = ((max_score_prob - min_score_prob)*(i+1))/(self.field_length - 1) + min_score_prob 
            elif i == self.field_length:
                self.mdp[i][1].append((1, self.field_length, 0, True))
            else:
                self.mdp[i][1].append((1, self.field_length + 1, 0, True))


    def reset(self):
        self.laststate = None
        self.state = 0
        
        return self.state
    
    def step(self, action):
        self.laststate = self.state
    
        transitions = self.mdp[self.laststate][action]
        index = categorical_sample([i[0] for i in transitions], np.random)

        prob, new_state, reward, termination = transitions[index]
        self.state = new_state
        
        return new_state, reward, termination

setattr(Environment, "render", render)
