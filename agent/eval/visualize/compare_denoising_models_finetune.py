import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from util.dirs import REINFLOW_DIR
import os

# Set font sizes
env_name = 'walker2d'
n_parallel_envs = 40
n_rollout_steps = 500
n_act_steps = 4  # Not used in sample calculation
plot_x_axis = 'sample'
output_dir = os.path.join(REINFLOW_DIR, f'visualize/ModelCompare/{env_name}')
data_path = f'visualize/ModelCompare/walker-ddpm-ddim-reflow-shortcut-correct.csv'

plt.rcParams.update({
    'axes.labelsize': 24,  # Axis labels
    'xtick.labelsize': 24,  # X-axis ticks
    'ytick.labelsize': 24,  # Y-axis ticks
    'legend.fontsize': 21   # Legend
})



# Define color schemes
red_to_purple_colors = [
    '#FF0000',  # Red
    '#CC0033',  # Red-Violet 
    '#990066'   # Dark Magenta
]
velvet_color = '#8B008B'  # Velvet (magenta/purple hue)
orange_color = '#FF8C00'  # Orange for DDPM
dashed_orange_color = '#FF8C00'  # Same as DDPM but dashed for DDIM

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(data_path)

# Function to parse column names and extract method and seed
def parse_column_name(col):
    pattern = r'(?:.*)_ppo_(diffusion_mlp_ta4_td20_tdf10|diffusion_DDIM_.*|reflow_mlp.*|shortcut_mlp.*)_seed(\d+).*'
    match = re.search(pattern, col)
    if match:
        method_full, seed = match.groups()
        if 'diffusion_mlp_ta4_td20_tdf10' in method_full:
            method = 'DDPM'
        elif 'diffusion_DDIM' in method_full:
            method = 'DDIM'
        elif 'reflow_mlp' in method_full:
            method = '1-ReFlow'
        else:  # shortcut_mlp
            method = 'ShortCut'
        return {
            'method': method,
            'seed': int(seed)
        }
    return None

# Extract metadata from column names
columns_info = {}
for col in df.columns:
    if col != 'Step' and ('eval/avg episode reward' in col or 'avg episode reward - eval' in col) and not ('__MIN' in col or '__MAX' in col):
        info = parse_column_name(col)
        if info:
            columns_info[col] = info

# Group data by method
method_data = {
    'DDPM': [],
    'DDIM': [],
    '1-ReFlow': [],
    'ShortCut': []
}

for col, info in columns_info.items():
    rewards = df[col]
    method_data[info['method']].append(rewards)

# Print number of seeds per method for verification
for method, rewards_list in method_data.items():
    print(f"{method}: {len(rewards_list)} seeds")

# Function to compute mean and std, handling varying lengths
def compute_stats(data_list):
    min_length = min(len(d) for d in data_list)
    truncated_data = [d[:min_length] for d in data_list]
    mean = np.mean(truncated_data, axis=0)
    std = np.std(truncated_data, axis=0)
    return mean, std, min_length

# Plot configuration
plt.figure(figsize=(10, 6))
plot_x_axis_candidate = ['step', 'sample']

# Compute x-axis
steps = df['Step']
if plot_x_axis == 'step':
    x_axis = steps
    plot_x_axis_name = 'Steps'
elif plot_x_axis == 'sample':
    x_axis = steps * n_parallel_envs * n_rollout_steps * n_act_steps
    plot_x_axis_name = 'Samples'
else:
    raise ValueError(f"Unsupported plot_x_axis type {plot_x_axis}. Must be {plot_x_axis_candidate}")

# Define the desired order for plotting and legend
method_order = ['ShortCut', '1-ReFlow', 'DDPM', 'DDIM']

# Plot each method in the specified order
method_styles = {
    'ShortCut': {'color': red_to_purple_colors[0], 'linestyle': '-'},
    '1-ReFlow': {'color': velvet_color, 'linestyle': '-'},
    'DDPM': {'color': orange_color, 'linestyle': '-'},
    # 'DDIM': {'color': dashed_orange_color, 'linestyle': '--'}
    'DDIM': {'color': 'brown', 'linestyle': '-'}
}

handles = []
labels = []
for method in method_order:
    rewards_list = method_data.get(method, [])
    if rewards_list:  # Only plot if there is data
        mean, std, min_length = compute_stats(rewards_list)
        x_axis_truncated = x_axis[:min_length]
        style = method_styles[method]
        line, = plt.plot(x_axis_truncated, mean, 
                        label=method, 
                        color=style['color'], 
                        linestyle=style['linestyle'], 
                        linewidth=3.0)
        plt.fill_between(x_axis_truncated, mean - std, mean + std, 
                        color=style['color'], 
                        alpha=0.2)
        handles.append(line)
        labels.append(method)

# Customize plot
if plot_x_axis == 'sample':
    plt.xlim(right=7e7)
# plt.title(f'Model Comparison (Walker2d, {env_name.capitalize()}, 4 steps)')
plt.xlabel(plot_x_axis_name)
plt.ylabel('Average Episode Reward')
plt.tick_params(axis='both', labelsize=24)  # Set tick label fontsize to 24
# plt.legend(handles, labels, loc='lower center')  # Legend in bottom left with specified order
plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.30, 0.0))
plt.grid(True)

# Save plot
output_path = os.path.join(output_dir, 'model_comparison')
plt.savefig(output_path + '.png', bbox_inches='tight')
plt.savefig(output_path + '.pdf', bbox_inches='tight')
plt.close()
print(f"Figure saved to {output_path + '.pdf'}")