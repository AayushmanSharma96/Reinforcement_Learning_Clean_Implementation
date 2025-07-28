import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import time

def reward_function(observation, action):
    diag_q = [1,10,1,1]; 
    r = 1;
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -cost

#env = gym.make("CartPole-v1", render_mode = "human") #"CartPole-v1" | "Swimmer-v4"
env = gym.make('InvertedPendulum-v4',render_mode = "human")
min_action = -20
max_action = 20
env = RescaleAction(env, min_action=min_action, max_action=max_action)
obs, _ = env.reset()
#print('obs = ', obs.dtype)
#print('env.observation_space=', env.observation_space, ' env.action_space = ', env.action_space)

#new_env_obs_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float64)
#new_env_action_space = gym.spaces.Box(-20, 20, (1,), np.float32)
#print('new_env_obs_space=', new_env_obs_space, ' new_env_action_space = ', new_env_action_space)

#env.observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float64)

#print('env.observation_space=', env.observation_space,)
q_pos = np.array([0,0])
q_vel = np.array([0,0])
env.set_state(q_pos, q_vel)
observation, reward, termination, truncated, info = env.step(0)
print("step:", 0, "observation:", observation," reward:", reward, " termination", termination, " truncated:", truncated)

for i in range(100):
    action =  env.action_space.sample()
    print("action:", action)
    
    observation, reward, termination, truncated, info = env.step(action)
    reward = reward_function(observation, action)
    
    env.render()

    time.sleep(1)
    print("step:", i, "observation:", observation," reward:", reward, " termination", termination, " truncated:", truncated)
    if abs(observation[0])>= 10:
        observation, info = env.reset()
    #     env.set_state(q_pos, q_vel)

env.close()