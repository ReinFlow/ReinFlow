# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from util.dirs import REINFLOW_DIR
import os

# Set font sizes
env_name='ant'
n_parallel_envs=40
n_rollout_steps=500
n_act_steps=4
plot_x_axis='sample'
output_dir = os.path.join(REINFLOW_DIR, f'visualize/Sensitivity/{env_name}')
data_path = f'visualize/Sensitivity/{env_name}/ant_sensitivity_sigma.csv'

line_width=3.0
plt.rcParams.update({
    'axes.labelsize': 24,  # Axis labels
    'xtick.labelsize': 24,  # X-axis ticks
    'ytick.labelsize': 24,  # Y-axis ticks
    'legend.fontsize': 21   # Legend
})
# Define purple color palette
# purple_colors = [
#     '#4B0082',  # Indigo
#     '#6A0DAD',  # Purple
#     '#800080',  # Purple
#     '#9932CC',  # Dark Orchid
#     '#BA55D3',  # Medium Orchid
#     '#DDA0DD'   # Plum
# ]
red_to_purple_colors = [
    '#FF0000',  # Red  # sigma(s,t)
    '#CC0033',  # Red-Violet
    '#990066',  # Dark Magenta  # sigma(s)
    '#660099',  # Purple
    "#2D2732"   # Indigo
]


# Load the CSV file
df = pd.read_csv(data_path)  
plot_x_axis_candidate=['sample', 'step']

# Function to parse column names and extract relevant info
def parse_column_name(col):
    pattern = r'sigma_(const|learnst|learns)_([\d\.]+)(?:_([\d\.]+))?_seed_(\d+)(?:_ent([\d\.]+))?'
    match = re.search(pattern, col)
    if match:
        noise_type, noise_level1, noise_level2, seed, ent_coef = match.groups()
        return {
            'noise_type': noise_type,
            'noise_level': float(noise_level1),
            'noise_level2': float(noise_level2) if noise_level2 else None,
            'seed': int(seed),
            'ent_coef': float(ent_coef) if ent_coef else 0.03  # Default to 0.03 if not specified
        }
    return None

# Extract metadata from column names
columns_info = {}
for col in df.columns:
    if col != 'Step' and 'eval/avg episode reward' in col and not ('__MIN' in col or '__MAX' in col):
        info = parse_column_name(col)
        if info:
            columns_info[col] = info

# Group data for the three figures
constant_noise_data = {}
noise_param_data = {}
entropy_data = {}

# Track sigma(s,t) columns
sigma_st_columns = []
sigma_s_columns = []

for col, info in columns_info.items():
    steps = df['Step']
    rewards = df[col]
    if info['noise_type'] == 'const':
        key = info['noise_level']
        if key not in constant_noise_data:
            constant_noise_data[key] = []
        constant_noise_data[key].append(rewards)
    elif info['noise_type'] in ['learnst', 'learns']:
        if info['noise_type'] == 'learnst' and info['ent_coef'] == 0.01:
            key = 'sigma(s,t)'
            if key not in noise_param_data:
                noise_param_data[key] = []
            noise_param_data[key].append(rewards)
            print(f"Append {key}: seed={info['seed']}, ent={info['ent_coef']:.2f}")
            sigma_st_columns.append(col)  # Track sigma(s,t) columns
            
        elif info['noise_type'] == 'learns':
            key = 'sigma(s)'
            if key not in noise_param_data:
                noise_param_data[key] = []
            noise_param_data[key].append(rewards)
            sigma_s_columns.append(col)
            print(f"Append {key}: seed={info['seed']}, ent={info['ent_coef']:.2f}")
    if info['noise_type'] == 'learnst':
        key = info['ent_coef']
        if key not in entropy_data:
            entropy_data[key] = []
        entropy_data[key].append(rewards)


# Print sigma(s,t) columns to verify seeds
print("\nColumns for sigma(s,t):")
for col in sigma_st_columns:
    print(col)
print(f"Number of seeds for sigma(s,t): {len(sigma_st_columns)}")

# Print sigma(s) columns to verify seeds
print("\nColumns for sigma(s):")
for col in sigma_s_columns:
    print(col)
print(f"Number of seeds for sigma(s): {len(sigma_s_columns)}")

print(f"noise_param_data={noise_param_data.keys()}")
for key, value in noise_param_data.items():
    print(f"key={key}, len={len(value)}")


# Function to compute mean and std, handling varying lengths
def compute_stats(data_list):
    min_length = min(len(d) for d in data_list)
    truncated_data = [d[:min_length] for d in data_list]
    mean = np.mean(truncated_data, axis=0)
    std = np.std(truncated_data, axis=0)
    return mean, std, min_length

# Plot 1: Constant Noise Level
plt.figure(figsize=(10, 6))
legend_order = [0.04, 0.001, 0.08, 0.20, 0.12, 0.16]
color_idx = 0
handles = []
labels = []

########################################################################
first_step_values=[]
########################################################################
for noise_level in legend_order:
    if noise_level in constant_noise_data:
        rewards_list = constant_noise_data[noise_level]
        mean, std, min_length = compute_stats(rewards_list)
        steps = df['Step'][:min_length]
        n_samples=steps * n_parallel_envs* n_rollout_steps * n_act_steps            
        if plot_x_axis=='step':
            x_axis=steps
            plot_x_axis_name='Steps'
        elif plot_x_axis=='sample':
            x_axis=n_samples
            plot_x_axis_name='Samples'
        else:
            raise ValueError(f"Unsupported plot_x_axis type {plot_x_axis}. plot_x_axis must be {plot_x_axis_candidate}")
        line, = plt.plot(x_axis, mean, color=red_to_purple_colors[color_idx], label=f'std={noise_level:.3f}', linewidth=line_width)
        plt.fill_between(x_axis, mean - std, mean + std, color=red_to_purple_colors[color_idx], alpha=0.2)
        handles.append(line)
        labels.append(f'std={noise_level:.3f}')
        color_idx = (color_idx + 1) % len(red_to_purple_colors)
        ########################################################################
        first_step_values.append(mean[0])
bc_value=np.array(first_step_values).mean()
########################################################################
# Add BC horizontal line
bc_line= plt.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=line_width)
handles.append(bc_line)
labels.append('BC')
########################################################################

# plt.title(f'Constant Noise Level (Entropy=0.03, {env_name})')
plt.xlabel(plot_x_axis_name)
plt.ylabel('Average Episode Reward')

# plt.legend(handles[::-1], labels[::-1], fontsize=18)  # Reverse legend order 
# display legend in boldface.
legend = plt.legend(handles[::-1], labels[::-1], fontsize=18)
for text in legend.get_texts():
    text.set_fontweight('bold')
# plt.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels[::-1]), fontsize=21) # horizontal legend.

plt.grid(True)
plt.gca().xaxis.get_offset_text().set_fontsize(24)  # Enlarge x-axis offset text (e.g., "1e7")
constant_noise_level_path = os.path.join(output_dir, 'constant_noise_level')
plt.savefig(constant_noise_level_path + '.png')
plt.savefig(constant_noise_level_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"\n\nConstant Noise Level Figure saved to {constant_noise_level_path + '.pdf'}\n\n")

# Initialize figure
plt.figure(figsize=(10, 6))

# Lists to store plot handles and labels
handles = []
labels = []

# Plot data
for noise_type, rewards_list in noise_param_data.items():
    print(f"rewards_list={len(rewards_list)}")
    mean, std, min_length = compute_stats(rewards_list)
    steps = df['Step'][:min_length]
    n_samples = steps * n_parallel_envs * n_rollout_steps * n_act_steps
    
    if plot_x_axis == 'step':
        x_axis = steps
        plot_x_axis_name = 'Steps'
    elif plot_x_axis == 'sample':
        x_axis = n_samples
        plot_x_axis_name = 'Samples'
    else:
        raise ValueError(f"Unsupported plot_x_axis type {plot_x_axis}. plot_x_axis must be {plot_x_axis_candidate}")
    
    color = red_to_purple_colors[0] if noise_type == 'sigma(s,t)' else red_to_purple_colors[2]
    if noise_type == 'sigma(s,t)':
        noise_type_label = r'ReinFlow-R: $\sigma(s,t)$'
    elif noise_type == 'sigma(s)':
        noise_type_label = r'ReinFlow-R: $\sigma(s)$'
    
    # Plot line and fill, store handle
    line, = plt.plot(x_axis, mean, label=noise_type_label, color=color, linewidth=line_width)
    plt.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)
    
    # Store handle and label
    handles.append(line)
    labels.append(noise_type_label)    
########################################################################
# Add BC horizontal line
bc_line= plt.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=line_width)
handles.append(bc_line)
labels.append('BC')
########################################################################
# Debugging: Print original handles and labels
print(f"Original handles={handles}, labels={labels}")
# Explicitly sort handles and labels for desired legend order (top to bottom: ReinFlow-R: $\sigma(s,t)$, ReinFlow-R: $\sigma(s)$)
desired_order = [r'ReinFlow-R: $\sigma(s,t)$', r'ReinFlow-R: $\sigma(s)$', r'BC']
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
sorted_handles, sorted_labels = zip(*sorted_handles_labels)
# Debugging: Print reordered labels
print(f"Reordered labels={sorted_labels}")

# Create legend with reordered handles and labels
plt.legend(sorted_handles, sorted_labels, loc='best')
# plt.title(f'Noise Parameterization (Entropy=0.01, {env_name})')
plt.gca().xaxis.get_offset_text().set_fontsize(24)  # Enlarge x-axis offset text (e.g., "1e7")
plt.xlabel(plot_x_axis_name)
plt.ylabel('Average Episode Reward')
# plt.legend()
plt.grid(True)
noise_param_path = os.path.join(output_dir, 'noise_parameterization')
plt.savefig(noise_param_path + '.png')
plt.savefig(noise_param_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"\n\nNoise Condition Figure saved to {noise_param_path + '.pdf'}\n\n")

# Plot 3: Entropy Level
plt.figure(figsize=(10, 6))
legend_order = sorted(entropy_data.keys(), reverse=True)  # Reverse order: 0.10 to 0.00
color_idx = 0
handles = []
labels = []
for ent_coef in legend_order:
    rewards_list = entropy_data[ent_coef]
    mean, std, min_length = compute_stats(rewards_list)
    steps = df['Step'][:min_length]
    n_samples=steps * n_parallel_envs* n_rollout_steps * n_act_steps
    if plot_x_axis=='step':
        x_axis=steps
        plot_x_axis_name='Steps'
    elif plot_x_axis=='sample':
        x_axis=n_samples
        plot_x_axis_name='Samples'
    else:
        raise ValueError(f"Unsupported plot_x_axis type {plot_x_axis}. plot_x_axis must be {plot_x_axis_candidate}")
    line, = plt.plot(x_axis, mean, color=red_to_purple_colors[color_idx], label=f'$\\alpha={ent_coef:.2f}$', linewidth=line_width)
    plt.fill_between(x_axis, mean - std, mean + std, color=red_to_purple_colors[color_idx], alpha=0.2)
    handles.append(line)
    labels.append(f'$\\alpha={ent_coef:.2f}$')
    color_idx = (color_idx + 1) % len(red_to_purple_colors)
########################################################################
# Add BC horizontal line
bc_line= plt.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=line_width)
handles.append(bc_line)
labels.append('BC')
########################################################################
# plt.title(f'Entropy Level (Sigma(s,t), Noise=0.08-0.16, {env_name})')
plt.gca().xaxis.get_offset_text().set_fontsize(24)  # Enlarge x-axis offset text (e.g., "1e7")
plt.xlabel(plot_x_axis_name)
plt.ylabel('Average Episode Reward')
plt.legend(handles, labels)  # Already reversed due to legend_order
plt.grid(True)
entropy_level_path = os.path.join(output_dir, 'entropy_level')
plt.savefig(entropy_level_path + '.png')
plt.savefig(entropy_level_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"\n\nEntropy Level Figure saved to {entropy_level_path + '.pdf'}\n\n")