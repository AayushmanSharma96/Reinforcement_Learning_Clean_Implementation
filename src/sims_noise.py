import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import numpy as np
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

ENV_NAME = 'InvertedPendulum-v4'

csv_file = '../data/sac_cartpole/sac_cartpole_monte_carlo_epsi100_v1.csv'
exp_name = 'sac_cartpole_ep_30_v1_epsi100'
run_name = 'sac'

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

LOG_STD_MAX = 2
LOG_STD_MIN = -5
MAX_OPEN_LOOP_CONTROL = 2.5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_mean = nn.Linear(512, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(512, np.prod(env.action_space.shape))
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

        #print(f"log_std = {log_std.exp()}")
        std = log_std.exp() 
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #x_t = mean #makeing the action deterministic
        y_t = torch.tanh(x_t)
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
        env = gym.make('InvertedPendulum-v4',render_mode = "rgb_array")
        env = gym.wrappers.RecordVideo(env, f"../videos/{run_name}")

    elif render_bool: 
        env = gym.make('InvertedPendulum-v4',render_mode = "human")

    else:
        env = gym.make('InvertedPendulum-v4')

    min_action = -30
    max_action = 30
    env = RescaleAction(env, min_action=min_action, max_action=max_action)
    env.reset()

    return env

def reward_function(observation, action):
    diag_q = [10,40,10,10]; 
    r = 0.5;
    #print("observation:", observation)
    #print("observation:", observation[0,1])
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2) +\
                r*(action**2)

    return -cost

def terminal_cost(observation):

    diag_q = [1000,4000,1000,1000]; 
    
    cost = diag_q[0]*(observation[0]**2) + diag_q[1]*(observation[1]**2) +\
                diag_q[2]*(observation[2]**2) + diag_q[3]*(observation[3]**2)

    return -cost

# Add this function to save mean and variance of cost and error
def save_to_csv(epsilon, cost_mean, cost_var, error_mean, error_var, csv_file):
    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'epsilon': [epsilon],
        'cost_mean': [cost_mean],
        'cost_variance': [cost_var],
        'error_mean': [error_mean],
        'error_variance': [error_var]
    })

    # Append the data to the CSV file
    df.to_csv(csv_file, mode='a', index=False, header=not pd.io.common.file_exists(csv_file))

def remove_upper_percentile_outliers(data, upper_percentile=95):
    upper_bound = np.percentile(data, upper_percentile)
    return data[data <= upper_bound]

def run_sample(epsilon, iter, actor, qf1):

    given_seed = iter
    total_timesteps = 30

    random.seed(given_seed)
    np.random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.backends.cudnn.deterministic = True
    obs, _ = env.reset(seed=given_seed)
    cost = 0

    for global_step in range(total_timesteps):
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            cost_to_go = -qf1(torch.Tensor(obs).to(device), actions).item()
            actions = actions.cpu().numpy().clip(env.action_space.low, env.action_space.high)

            #adding control noise
            w = epsilon*np.random.normal(0.0,1.0)*MAX_OPEN_LOOP_CONTROL
            actions = actions + w
            actions = actions.clip(env.action_space.low, env.action_space.high)

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        rewards = reward_function(obs, actions)
        cost -=rewards
        #print(f"i = {global_step} cost = {rewards}")
        if terminations:
            obs, _ = env.reset()
            # env.set_state(q_pos, q_vel)
            # obs, rewards, terminations, truncations, infos = env.step(0.0)
            
        #env.render()
        
        #print("observation:", obs, " action:", actions, ' CTG=', cost_to_go, ' w = ', w)
        
        obs = next_obs
    
    cost -= terminal_cost(obs)
    obs[1] = angle_normalize(obs[1]) #normalize theta between -pi to pi
    error = np.linalg.norm(obs)
    #print(f"cost = {cost} error = {error}")
    return cost[0], error

def angle_normalize(x):
    '''
    Function to normalize the pendulum's angle from [0, Inf] to [-np.pi, np.pi]
    '''
    
    return -((-x+np.pi) % (2*np.pi)) + np.pi

if __name__ == "__main__":
    
    # if GPU is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}");

    env = make_env(ENV_NAME, render_bool = False, record_video=False)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    print(f"loading ../runs/{run_name}/{exp_name}.pth")
    checkpoint = torch.load(f"../runs/{run_name}/{exp_name}.pth")
    actor.load_state_dict(checkpoint[0])
    qf1.load_state_dict(checkpoint[1])

    actor.eval()
    qf1.eval() 

    epsi_range = np.linspace(1.2,1.8,4)
    #epsi_range = np.append(epsi_range, np.linspace(2,5,4), axis=0)
    #epsi_range = np.linspace(0,0,1)
    mc_runs = 100
    

    print("Epsilon range:", epsi_range)

    # Clear the CSV file before running (optional)
    
    with open(csv_file, 'a') as f:
        f.write('epsilon,cost_mean,cost_variance,error_mean,error_variance\n')

    for epsilon in epsi_range:

        cost = np.zeros(mc_runs)
        error = np.zeros(mc_runs)

        for iter in range(mc_runs):

            cost_iter, error_iter = run_sample(epsilon, iter, actor, qf1)
            cost[iter] = cost_iter
            error[iter] = error_iter

        # Calculate the mean and variance of cost and error
        cost_filtered = remove_upper_percentile_outliers(cost, upper_percentile=95)
        cost_mean = np.mean(cost_filtered)
        cost_var = np.var(cost_filtered)

        error_filtered = remove_upper_percentile_outliers(error, upper_percentile=95)
        error_mean = np.mean(error_filtered)
        error_var = np.var(error_filtered)

        print(f"epsilon = {epsilon}, cost_mean = {cost_mean}, cost_var = {cost_var}, \
              error_mean = {error_mean}, error_var = {error_var}")
        # Save the results to a CSV file
        save_to_csv(epsilon, cost_mean, cost_var, error_mean, error_var, csv_file)

        

    env.close()
    