from env import Environment
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use("Agg")

# Modes for rendering:
metadata = {
    "render_modes": ["human", "rgb_array", None],
    "render_fps": 4,
}

def evaluate(env, policy, file, num_runs=5):
    """ Evaluates the environment based on a policy.

    Please use this method to debug your code for the environment.

    Args:
        env: Environment we want to use. 
        policy: Numpy array of shape (num_states, num_actions), for each state the array contains
            the probabilities of entering the successor state based on the associated action. 
        file: File used for storing the video.
        num_runs: Number of runs displayed.
    """
    
    frames = []  # collect rgb_image of agent env interaction
    video_created = False
    for _ in range(num_runs):
        done = False
        obs = env.reset()
        while not done:
            action =  np.random.choice(np.flatnonzero(np.isclose(policy[obs], max(policy[obs]), rtol=0.0001)))
            out = env.render()
            frames.append(out)
            obs, reward, done = env.step(action)
            if done:
                out = env.render()
                frames.append(out)
                
    # create animation out of saved frames
    if all(frame is not None for frame in frames):
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        img = plt.imshow(frames[0][0])
        def animate(index):
            img.set_data(frames[index][0])
            return [img]
        anim = FuncAnimation(fig, animate, frames=len(frames), interval=20)
        plt.close()
        anim.save(file, writer="ffmpeg", fps=2)
        video_created = True
        
        
        
environment = Environment(min_score_prob = 0.0, max_score_prob = 0.8, line_position = 2, field_length = 10, render_mode="rgb_array")
policy = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]]) # probabilies for actions per state
video_file_1 = "Policy_Iteration/basketball_debug.mp4"
evaluate(environment, policy, video_file_1)
