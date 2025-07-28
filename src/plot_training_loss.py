import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

params = {'axes.labelsize':14,
            'font.size':14,
            'legend.fontsize':14,
            'xtick.labelsize':12,
            'ytick.labelsize':12,
            'text.usetex':False,
            'figure.figsize':[12,8]}
plt.rcParams.update(params)
# Set font type to Type 42 (TrueType) for embedding in PDF
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Load the data from the CSV file
path = "."#'../data/ddpg_cartpole/'
file_name_1 = 'sac_swimmer_output_v5_buffer10_5_5M_1'
file_name_2 = 'sac_swimmer_output_v5_buffer10_5_5M_2'
file_name_3 = 'sac_swimmer_output_v5_buffer10_5_5M_4'
file_name_4 = 'sac_swimmer_output_v5_buffer10_5_5M_5'
# file_name_2 = 'sac_cartpole_output_v4_buffer10_3_1M_2'
# file_name_3 = 'sac_cartpole_output_v4_buffer10_3_1M_3'
# file_name_4 = 'sac_cartpole_output_v4_buffer10_7_5M_4'
# file_name_5 = 'sac_cartpole_output_v4_buffer10_3_1M_5'
file_path_1 = f"{path}/{file_name_1}.csv"  # Replace with your actual file path
file_path_2 = f"{path}/{file_name_2}.csv"
file_path_3 = f"{path}/{file_name_3}.csv"
file_path_4 = f"{path}/{file_name_4}.csv"
# file_path_5 = f"{path}/{file_name_5}.csv"
buffer_size = '1e5'
data_1 = pd.read_csv(file_path_1)
data_2 = pd.read_csv(file_path_2)
data_3 = pd.read_csv(file_path_3)
data_4 = pd.read_csv(file_path_4)
# data_5 = pd.read_csv(file_path_5)

# Plotting
plt.figure()

cost_vec_1 = []
cost_vec_2 = []
cost_vec_3 = []
cost_vec_4 = []
# cost_vec_5 = []

for cost in data_1['cost']:
    # cost_vec_1.append(float(cost.strip('[]')))
    cost_vec_1.append(float(cost))
for cost in data_2['cost']:
    cost_vec_2.append(float(cost))
#     cost_vec_2.append(float(cost.strip('[]')))
for cost in data_3['cost']:
    cost_vec_3.append(float(cost))
#     cost_vec_3.append(float(cost.strip('[]')))
for cost in data_4['cost']:
    cost_vec_4.append(float(cost))
#     cost_vec_4.append(float(cost.strip('[]')))
# for cost in data_5['cost']:
#     cost_vec_5.append(float(cost.strip('[]')))

len_episodes_1 = np.arange(1,len(data_1['step'])*10,10)
# print(np.shape(len_episodes_1[:150]))
# print(np.shape(cost_vec_1[:150]))
len_episodes_2 = np.arange(1,len(data_2['step'])*10,10)
len_episodes_3 = np.arange(1,len(data_3['step'])*10,10)
len_episodes_4 = np.arange(1,len(data_4['step'])*10,10)
# len_episodes_5 = np.arange(1,len(data_5['step'])*100,100)

# Plot episode cost
# plt.subplot(2, 1, 1)
plt.plot(len_episodes_1[:150], cost_vec_1[:150], color='blue', linewidth = 3, label='Episode Cost for Instance 1')
# plt.plot(len_episodes_1[:150], cost_vec_2[:150], color='red', linewidth = 3, label='Episode Cost for Buffer = 10^6')
# plt.plot(len_episodes_1[:150], cost_vec_3[:150], color='green', linewidth = 3, label='Episode Cost for Buffer = 10^7')
plt.plot(len_episodes_2[:150], cost_vec_2[:150], color='green', linewidth = 3, label='Episode Cost for Instance 2')
plt.plot(len_episodes_3[:150], cost_vec_3[:150], color='red', linewidth = 3, label='Episode Cost for Instance 3')
plt.plot(len_episodes_4[:150], cost_vec_4[:150], color='black', linewidth = 3, label='Episode Cost for Instance 4')
# plt.plot(len_episodes_5, cost_vec_5, color='orange', linewidth = 3, label='Episode Cost for Instance 5')
plt.xlabel('Rollouts')
plt.ylabel('Episode Cost')
plt.title('Swimmer: SAC Training Metrics Buffer size = '+ buffer_size)
plt.grid()
plt.legend()

#Plot Q-function loss (qf1_loss)
# plt.subplot(2, 1, 1)
# plt.plot(len_episodes_1, data_1['qf_loss'], color='blue',linewidth = 3, label='Critic Loss for Instance 1')
# plt.plot(len_episodes_2, data_2['qf_loss'], color='green',linewidth = 3, label='Critic Loss for Instance 2')
# plt.plot(len_episodes_3, data_3['qf_loss'], color='red',linewidth = 3, label='Critic Loss for Instance 3')
# plt.plot(len_episodes_4, data_4['qf_loss'], color='black',linewidth = 3, label='Critic Loss for Instance 4')
# plt.plot(len_episodes_5, data_5['qf_loss'], color='orange',linewidth = 3, label='Critic Loss for Instance 5')
# #plt.yscale('log')
# plt.xlabel('Rollouts')
# plt.ylabel('QF Loss')
# plt.grid()
# plt.legend()

# # Plot actor loss
# plt.subplot(2, 1, 2)
# plt.plot(len_episodes_1, data_1['actor_loss'], color='blue',linewidth = 3, label='Actor Loss for Instance 1')
# plt.plot(len_episodes_2, data_2['actor_loss'], color='green',linewidth = 3, label='Actor Loss for Instance 2')
# plt.plot(len_episodes_3, data_3['actor_loss'], color='red',linewidth = 3, label='Actor Loss for Instance 3')
# plt.plot(len_episodes_4, data_4['actor_loss'], color='black',linewidth = 3, label='Actor Loss for Instance 4')
# plt.plot(len_episodes_5, data_5['actor_loss'], color='orange',linewidth = 3, label='Actor Loss for Instance 5')
# plt.xlabel('Rollouts')
# plt.ylabel('Actor Loss')
# plt.grid()
# plt.legend()

# Adjust layout
# plt.tight_layout()

# Save the plot as a PDF with embedded fonts
plt.savefig(f"../plots/SAC_losses_training_metrics_{file_name_1}_combined_sub.pdf", format='pdf', bbox_inches='tight')

# Display the plot (optional)
#plt.show()
