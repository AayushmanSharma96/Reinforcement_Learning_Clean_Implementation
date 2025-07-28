import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import time

ENV_NAME = 'InvertedPendulum-v4'
k = '_1'
# csv_file = 'sac_swimmer_output'+k+'.csv' #csv file to store training progress
exp_name = 'sac_swimmer_v5_buffer10_7_5M'+k
# run_name = 'sac'

# ENV_NAME = 'AttitudeControl-v0'#'Swimmer-v5'
k = 5
csv_file = 'sac_cartpole_output_v4_buffer10_3_1M_'+str(k)+'.csv'#'sac_swimmer_output_v5_buffer10_4_10M_10.csv'#'sac_cartpole_output_v4_buffer10_3_1M_'+str(k)+'.csv' #csv file to store training progress
exp_name = 'sac_cartpole_v4_buffer10_3_1M_'+str(k)
run_name = 'sac'

# k = 23
# csv_file = 'sac_attitude_output_v0_buffer10_6_1M_'+str(k)+'.csv' #csv file to store training progress
# exp_name = 'sac_attitude_v0_buffer10_6_1M_'+str(k)
# run_name = 'sac'

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 512)#400)
        self.fc2 = nn.Linear(512, 512)#nn.Linear(400, 300)
        self.fc3 = nn.Linear(512, 1)#nn.Linear(300, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

LOG_STD_MAX = 2
LOG_STD_MIN = -5
MAX_OPEN_LOOP_CONTROL = 4#2.5
epsilon = 0.0

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 512)#400)
        self.fc2 = nn.Linear(512,512)#(400, 300)
        self.fc_mean = nn.Linear(512, np.prod(env.action_space.shape))#(300, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(512, np.prod(env.action_space.shape))#(300, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)

        # print(f"log_std = {log_std.exp()}")
        std = log_std.exp() 
        normal = torch.distributions.Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        x_t = mean #makeing the action deterministic
        y_t = torch.tanh(x_t)
        #device = "cuda"
        #self.action_scale = torch.Tensor(np.array([100.0])).to(device)
        # print(f"action_scale = {self.action_scale}, action bias = {self.action_bias}")
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #print(f"x_t = {x_t} y_t = {y_t} action = {action} log_prob = {log_prob} log_prob.shape = {log_prob.shape}")
        log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def make_env(env_id, render_bool, record_video=False):

    if record_video:
        env = gym.make(env_id,render_mode = "rgb_array")
        env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}")

    elif render_bool: 
        env = gym.make(env_id,render_mode = "human")

    else:
        env = gym.make(env_id)

    min_action = -100
    max_action = 100
    # env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env




# Saving the actions to a CSV file
def save_actions_to_csv(action_vec, filename="actions.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Action"])  # Write header
        for action in action_vec:
            writer.writerow([action])

def reward_function(observation, action):
    # diag_q = [1,1];#,1,1]; 
    # r = 0.0001;
    # dt = 0.01
    
    
    # diag_q[2]*(x_velocity**2)+diag_q[3]*(y_velocity**2)+
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    # cost = diag_q[0]*((observation[0]-0.6)**2) + diag_q[1]*((observation[1]+0.5)**2) +\
    #             r*(np.square(action).sum())
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    # cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
    #             diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
    #             r*(action**2)
    # cost = diag_q[0]*((observation[0]-0.6)**2) + diag_q[1]*((observation[1]+0.5)**2) +\
    #             r*(np.square(action).sum())
    # diag_q = 1000*np.array([1,1,1,1,1,1]) 
    # r = 1*np.array([1,1,1])
    dt = 0.1
    
    a = 1
    b = 3
    
    ###################### CARTPOLE ##############################
    diag_q = [1,10,1,1]; 
    r = 1;
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)
    ##############################################################
    
    # cost_fwd = 15 * ((observation[0])**2 + (observation[1] - 0.4)**2 + (observation[2] - 0.2)**2) \
    #                  + 0.5 * (np.square(observation[4:7]).sum() + (observation[3] - 1)**2)
        
    # Compute control cost
    # cost_ctrl = r * np.square(action).sum()
   
    # cost = diag_q[0]*observation[0]*observation[0]+diag_q[1]*observation[1]*observation[1]+diag_q[2]*observation[2]*observation[2]+diag_q[3]*observation[3]*observation[3]+diag_q[4]*observation[4]*observation[4]+diag_q[5]*observation[5]*observation[5]
    
    ctrl_cost = 0#r[0]*np.square(action[0])+r[1]*np.square(action[1])+r[2]*np.square(action[2])
    return -(cost+ctrl_cost)
    



if __name__ == "__main__":

    given_seed = 1
    total_timesteps = 100
    gamma = 0.99

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = False, record_video=False)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    # checkpoint = torch.load(f"../runs/{run_name}/{exp_name}.pth") ## REMOVE THE _ROLLOUTS FOR REGULAR
    checkpoint = torch.load(f"../runs/sac/sac_cartpole_v1_buffer10_5.pth")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    actor.eval()
    qf1.eval() 

    obs, _ = env.reset(seed=10)
    q_pos = np.array([0,np.pi])
    q_vel = np.array([0,0])
    env.unwrapped.set_state(q_pos, q_vel)
    obs[0:2] = q_pos
    obs[2:] = q_vel
    #print(obs) 
    cost = 0
    
    
    ''' Generating Rollouts '''
    num_rollouts = 10#00#500
    rollout_length = 100  # Timesteps per rollout
    all_observations = []  # List to store all observations
    obs_init = []
    obs_success_init = []
    
    init_states = np.load('cartpole_init_2.npy')
    
    for rollout_idx in range(num_rollouts):
        obs, _ = env.reset()#seed=0)  # Reset the environment
        # q_pos = np.array([0.0+np.random.normal(0,5),np.pi+np.random.normal(0,0.5)])
        # q_vel = np.array([0,0])
        qpos = init_states[rollout_idx,:2]
        qvel = init_states[rollout_idx,2:]
        env.unwrapped.set_state(qpos, qvel)
        obs[:2] = qpos
        obs[2:] = qvel
        # env.state = np.concatenate([q_pos, q_vel])
        # Unwrap the RescaleAction wrapper to access the original environment
        # original_env = env.unwrapped
    
        # Now get the current observation from the unwrapped environment
        # obs = original_env._get_obs()

        # obs = env._get_obs()
        total_cost = 0  # Track cumulative cost for the rollout

        print(f"\nRollout {rollout_idx + 1}")
        print("Initial Step : ", obs)
        obs_i = obs
        obs_init.append(obs_i)
        for t in range(rollout_length):
            all_observations.append(obs)  # Save current observation

            # Get action from policy
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                w = epsilon*np.random.normal(0.0,1.0)*MAX_OPEN_LOOP_CONTROL
                actions = actions + w
                actions = actions.clip(env.action_space.low, env.action_space.high)

            # Step the environment
            next_obs, reward, done, truncation, info = env.step(actions)

            # Compute rollout cost using custom reward function
            total_cost += -reward_function(obs, actions)
            obs = next_obs

            # Print rollout status
            # print(f"Step {t}: Obs={obs}, Action={actions}, Reward={reward}, Total Cost={total_cost}")

            if done or truncation:
                break  # End the rollout if the environment is done
        print('final state : ', next_obs)
        if np.linalg.norm(next_obs,2)<0.5:
            obs_success_init.append(obs_i)
    # Convert observations list to numpy array and save
    # all_observations = np.array(all_observations)
    # output_file = "cartpole_observations.npy"
    # np.save(output_file, all_observations)
    # print(f"All observations saved to {output_file}")
    # output_init = "cartpole_init_2.npy"
    output_success_init = "cartpole_success_init_test.npy"
    # obs_init = np.array(obs_init)
    obs_success_init = np.array(obs_success_init)
    # np.save(output_init, obs_init)
    # np.save(output_success_init, obs_success_init)
    env.close()
    print("Rollouts complete.")
    
            
    # print(f'obs={obs}')
    # q_pos = np.array([-0.1,0.0])
    # q_vel = np.array([0,0])
    # env.set_state(q_pos, q_vel)
    # obs, rewards, terminations, truncations, infos = env.step(0.0)
    #print(f'obs={obs}')
    
    
    
    '''n_roll = 200
    norm_list = np.zeros((n_roll, 1))
    action_vec = []
    eps = np.linspace(0,1,11)
    
    det_actions = []
    
    obs, _ = env.reset(seed=100)
            
    ############ COMPUTE NOMINAL CONTROL SEQUENCE ##################################
    for global_step in range(total_timesteps):
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            
            cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
            actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            action_vec.append(actions[0])
            #adding control noise
            # w = epsilon*np.random.normal(0.0,1.0)*MAX_OPEN_LOOP_CONTROL
            # actions = actions + w
            # actions = actions.clip(env.action_space.low, env.action_space.high)
            det_actions.append(actions)

        rewards = reward_function(obs, actions)
        cost -=rewards
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if terminations:
            obs, _ = env.reset()
        
        obs = next_obs
    
    with torch.no_grad():
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
        actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
    ###########################################################################################
    print('Testing for open loop control: Cartpole')
    for epsilon in eps:
        for t in range(n_roll):
            obs, _ = env.reset(seed=100)
            
            for global_step in range(total_timesteps):
                with torch.no_grad():
                    # actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                    actions = torch.Tensor(det_actions[global_step]).to(device)
                    
                    cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
                    actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                    action_vec.append(actions[0])
                    #adding control noise
                    w = epsilon*np.random.normal(0.0,1.0)*MAX_OPEN_LOOP_CONTROL
                    actions = actions + w
                    actions = actions.clip(env.action_space.low, env.action_space.high)
        
                rewards = reward_function(obs, actions)
                cost -=rewards
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                if terminations:
                    obs, _ = env.reset()
                    # env.set_state(q_pos, q_vel)
                    # obs, rewards, terminations, truncations, infos = env.step(0.0)
                    
                # env.render()
                # time.sleep(0.1)
                # print("observation:", obs, " action:", actions, ' CTG=', cost_to_go, ' w = ', w)
                
                obs = next_obs
            
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            # print("observation:", obs, " action:", actions, ' CTG=', cost_to_go)
            # print("actions vec =", action_vec)
            #save_actions_to_csv(action_vec, filename="actions_sac_horizon_30.csv")
        
            # print(f"Final cost = {cost}")
            norm_list[t] = np.linalg.norm(obs, 2)
        print(epsilon, ' : Mean = ', np.mean(norm_list), ', std = ', np.std(norm_list))
    env.close()'''
    