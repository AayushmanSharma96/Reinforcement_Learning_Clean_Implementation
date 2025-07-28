import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from gymnasium.envs.registration import register
import numpy as np
import random
import time
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines3.common.buffers import ReplayBuffer
import csv

register(
    id='AttitudeControlEnv-v0',
    entry_point="gymnasium.envs.mujoco.attitude_control_v0:AttitudeControlEnv",
    max_episode_steps=2000,
)

ENV_NAME = 'AttitudeControlEnv-v0'
k = 25
csv_file = 'sac_attitude_output_v0_buffer10_6_1M_'+str(k)+'.csv' #csv file to store training progress
exp_name = 'sac_attitude_v0_buffer10_6_1M_'+str(k)
run_name = 'sac'

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mean = nn.Linear(300, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(300, np.prod(env.action_space.shape))
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
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        #print(f"x_t = {x_t} y_t = {y_t} action = {action} log_prob = {log_prob} log_prob.shape = {log_prob.shape}")
        # If the action space has more than one dimension, sum over the action dimension
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1, keepdim=True)  # Sum over action dimensions
        else:
            log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        
        return action, log_prob, mean

def reward_function(observation, action, next_obs):
    diag_q = 1e-3*1000*np.array([1,1,1,1,1,1]) 
    r = 1e-3*np.array([1,1,1])#0.000001*np.array([1,1,1,1000,1000,1000])
    dt = 0.1
    
    a = 1
    b = 3
    
    # cost_fwd = 15 * ((observation[0])**2 + (observation[1] - 0.4)**2 + (observation[2] - 0.2)**2) \
    #                  + 0.5 * (np.square(observation[4:7]).sum() + (observation[3] - 1)**2)
        
    # Compute control cost
    # cost_ctrl = r * np.square(action).sum()
   
    cost = diag_q[0]*observation[0]*observation[0]+diag_q[1]*observation[1]*observation[1]+diag_q[2]*observation[2]*observation[2]+diag_q[3]*observation[3]*observation[3]+diag_q[4]*observation[4]*observation[4]+diag_q[5]*observation[5]*observation[5]
    
    ctrl_cost = r[0]*np.square(action[0])+r[1]*np.square(action[1])+r[2]*np.square(action[2])
    return -(cost+ctrl_cost)

def make_env(env_id, render_bool):

    if render_bool:

        env = gym.make('AttitudeControlEnv-v0',render_mode = "human")
    else:
        env = gym.make('AttitudeControlEnv-v0')

    min_action = -20000
    max_action = 20000
    # env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

def write_data_csv(data):
    

    # Write the data to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writerow(['step', 'cost', 'qf_loss', 'actor_loss', 'observations', 'action'])
        
        # Write the data
        writer.writerow(data)
        
def save_observations_to_numpy(rb, filename="sac_replay_buffer_observations_10_3_", k="1"):
    # Extract the valid observations (up to the current buffer position)
    observations = rb.observations[: rb.pos]#.cpu().numpy()
    # Save as a .npy file
    np.save(filename+k+".npy", observations)
    print(f"Observations saved to {filename}")

if __name__ == "__main__":

    given_seed = np.random.randint(50)
    buffer_size = int(1e6)
    batch_size = 256
    total_timesteps = 1000000 #default = 1000000
    learning_starts = 25000 #default = 25e3
    episode_length = 2000
    exploration_noise = 0.0001
    policy_frequency = 2
    tau = 0.005 # weight to update the target network
    gamma = 0.95 #discount factor
    learning_rate = 3e-5
    alpha = 0.2 #Entropy regularization coefficient
    target_network_frequency = 1
    MAX_OPEN_LOOP_CONTROL = 2.5
    epsilon = 0.0


    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    
    #reward function parameters
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = False)
    
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    #
    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)

    # load pretrained model.
    # checkpoint = torch.load(f"runs/{run_name}/{exp_name}.pth")
    # actor.load_state_dict(checkpoint[0])
    # qf1.load_state_dict(checkpoint[1])
    # qf2.load_state_dict(checkpoint[2])

    #target network 
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)

    #initalizing target  with the same weights
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    #choose optimizer
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

    #experience replay buffer
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    episode_t = 0 
    episode_count = 0
    cost = 0
    obs, _ = env.reset()
    print(f'initial state = {obs}')
    
    for global_step in range(total_timesteps):
        
        if global_step < learning_starts:
            actions = np.array(env.action_space.sample())
            
        else:
            with torch.no_grad():
                #print(f"observation = {obs}")
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)
                
                #adding control noise
                w = epsilon*np.random.normal(0.0,1.0)*MAX_OPEN_LOOP_CONTROL
                actions = actions + w
                actions = actions.clip(env.action_space.low, env.action_space.high)

        
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        rewards = reward_function(obs, actions, next_obs)
        cost -=rewards 
        #print('step=', global_step, ' actions=', actions, ' rewards=', rewards,\
        #      ' obs=', next_obs, ' termination=', terminations, ' trunctions=', truncations)

    
        # save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        # if truncations:
        #     real_next_os = infos["final_observation"]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # reset observation
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:

            #sample experience from replay buffer
            data = rb.sample(batch_size)
            
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0:

                for _ in range(policy_frequency):
                    # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                
                if episode_count % 10 == 0:
                    print(f'step= {global_step} rewards= {rewards} qf_loss = {qf_loss.item()} '
                        f'actor_loss = {actor_loss.item()} observations= {obs} action= {actions}')

            # update the target networks
            if global_step % target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if global_step % 1000 == 0:
                
                print("SPS:", int(global_step / (time.time() - start_time)))
                # Data to write
                

        episode_t += 1
        
        if episode_t == episode_length:
            print('Observation : ', next_obs)
            print('resetting')
            episode_count += 1
            if episode_count % 10 == 0 and global_step > learning_starts:
                write_data = [global_step, cost, qf_loss.item(), actor_loss.item(), obs, actions]
                write_data_csv(write_data)
                
                # if global_step<30000:
                #     save_observations_to_numpy(rb, k="1")
                    
                # if global_step>80000 and global_step<90000:
                #     save_observations_to_numpy(rb, k="2")
                    
                # if global_step>680000 and global_step<700000:
                #     save_observations_to_numpy(rb, k="3")
                    
            obs, _ = env.reset()
            episode_t = 0
            print(f'Cost = {cost}')
            cost = 0
            
            


    save_model = True
    if save_model:
        model_path = f"../runs/{run_name}/{exp_name}.pth"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

    # save_observations_to_numpy(rb, k="4")
    env.close()
