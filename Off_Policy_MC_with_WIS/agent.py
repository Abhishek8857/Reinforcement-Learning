import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from render_util import plot_action_value
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")

class Agent:
    def __init__(self, env, gamma=0.9):
        """Initialises the environment and defines the dynamics"""
        self.env = env
        self.action_value_fn = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.C = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # Cumulative sum of ratios
        self.gamma = gamma
        
        def make_random_policy():
            def policy(state):
                bp = np.ones(self.env.action_space.n, dtype=float) / self.env.action_space.n
                return bp
            return policy
        
        self.behavior_policy = make_random_policy()
        
        
        def make_target_policy():
            def policy(state):
                pi = np.zeros(self.env.action_space.n, dtype=float)
                is_diff = np.isclose(self.action_value_fn[state], (self.action_value_fn[state].max()), rtol=0.01)
                index = np.flatnonzero(is_diff)
                
                if len(index) == 1:
                    pi[index[0]] = 1.0
                else:
                    pi[index] = 1.0 / len(index)
                    
                return pi
            return policy
        
        self.target_policy = make_target_policy()
        
        
    def get_action(self, policy, state):
        return np.random.choice(a=np.arange(self.env.action_space.n), p=policy(state))
    
    
    def update_return(self, gamma, discounted_return, reward):
        return (gamma * discounted_return) + reward


    def update_W(self, weight, t_pol, b_pol):
        return weight * (t_pol/b_pol)
    
    
    def train(self, num_episodes, max_duration=100):    
        for i in range(num_episodes + 1):
            episode = []
            state, info = self.env.reset()
            for j in range(max_duration):
                action = self.get_action(self.behavior_policy, state)
                next_state, reward, converged, truncate, info = self.env.step(action)
                episode.append((state, action, reward))
                
                if converged:
                    break
                
                state = next_state
            episode = np.array(episode)
            episode_duration = len(episode[:, :1])
            
            G = 0.0
            W = 1.0 
            
            for i in range(episode_duration - 1, -1, -1):
                state = int(episode[i][0])
                action = int(episode[i][1])
                reward = episode[i][2]
                
                G = self.update_return(self.gamma, G, reward)
                self.C[state][action] += W
                self.action_value_fn[state][action] += (W / self.C[state][action]) * (G - self.action_value_fn[state][action])
                W = self.update_W(W, self.target_policy(state)[action], self.behavior_policy(state)[action])
                
                if W == 0:
                    break
                
                
    def evaluate(self, env, file, num_runs=5):
        """ Evaluates the agent in the environment.

        Args:
            env: Environment we want to use. 
            file: File used for storing the video.
            num_runs: Number of runs displayed
        """
        frames = []  # collect rgb_image of agent env interaction
        video_created = False
        for _ in range(num_runs):
            done = False
            obs, info = env.reset()
            out = env.render()
            frames.append(out)
            while not done:
                action = self.get_action(self.target_policy, obs)
                obs, reward, done, truncated, info = env.step(action)
                out = env.render()
                frames.append(out)
        # create animation out of saved frames
        if all(frame is not None for frame in frames):
            fig = plt.figure(figsize=(10, 6))
            plt.axis('off')
            img = plt.imshow(frames[0])
            def animate(index):
                img.set_data(frames[index])
                return [img]
            anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
            plt.close()
            anim.save(file, writer="ffmpeg", fps=5) 
        return

    def plot_action_value(self):
        plot_action_value(self=self)
