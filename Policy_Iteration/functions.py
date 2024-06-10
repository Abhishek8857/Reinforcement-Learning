# import env
# import numpy as np
# from gymnasium.envs.toy_text.utils import categorical_sample


# def reset():
#     env.last_state = None
#     env.state = 0
    
#     return env.state


# def step(action):
#     env.laststate = env.state
    
#     transitions = env.mdp[env.laststate][action]
#     index = categorical_sample([i[0] for i in transitions], np.random)

#     prob, new_state, reward, termination = transitions[index]
#     env.state = new_state
    
#     return new_state, reward, termination

