import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import re
import hydra
from omegaconf import OmegaConf
from util.dirs import REINFLOW_DIR 

def extract_and_plot(evaluation_name, 
                     environment_name, 
                     task_name, 
                     re_expression,
                     csv_filename,
                     output_dir,
                     output_filename,
                     plot_x_axis,
                     n_parallel_envs,
                     n_rollout_steps,
                     n_act_steps):
    data = pd.read_csv(f'visualize/Final_experiments/data/finetune/{environment_name}/{task_name}/{csv_filename}')
    print(f"data['Step'].values={max(data['Step'].values)}")
    data = data[data['Step'] < 1000]
    eval_freq =10
    n_evals   = int(1000/eval_freq )
    print(f"Truncated data to Step <= 999. Will truncate eval data to below {n_evals} steps. ")
    
    # Calculate x-axis based on plot_x_axis
    if plot_x_axis == 'step':
        x_axis = data['Step'].values
        x_label = 'Steps'
    elif plot_x_axis == 'sample':
        steps = data['Step'].values
        x_axis = steps * n_parallel_envs * n_rollout_steps * n_act_steps
        x_label = 'Samples'
    else:
        raise ValueError(f"Unsupported plot_x_axis: {plot_x_axis}. Must be 'step' or 'sample'.")
    
    # Extract methods and seeds from column names
    method_seed_map = {}
    unmatched_columns = []
    
    for col in data.columns:
        if 'reward' in col and ('__MIN' not in col) and ('__MAX' not in col):
            print(f'col={col}')
            match = re.search(re_expression, col)
            if match:
                method = match.group(1)
                seed = int(match.group(2) or match.group(3) or match.group(4))
                if method not in method_seed_map:
                    method_seed_map[method] = []
                method_seed_map[method].append((seed, col))
            else:
                unmatched_columns.append(col)
    
    # Define method names and colors
    method_config = [
        {'original_name': 'ppo_shortcut_mlp_ta4_td4', 'display_name': 'ReinFlow-S (ours)', 'color': '#FF2400'},
        {'original_name': 'ppo_reflow_mlp_ta4_td4', 'display_name': 'ReinFlow-R (ours)', 'color': '#8B0000'},
        {'original_name': 'ppo_reflow_beta_mlp_ta4_td4', 'display_name': 'ReinFlow-beta (ours)', 'color': '#C000C0'},
        {'original_name': 'ppo_reflow_logitnormal_mlp_ta4_td4', 'display_name': 'ReinFlow-logitnormal (ours)', 'color': '#4B0082'},
        {'original_name': 'ppo_diffusion_mlp_ta4_td20_tdf10', 'display_name': 'DPPO', 'color': '#FFB300'},
        {'original_name': 'qsm_diffusion_mlp_ta4_td20', 'display_name': 'QSM', 'color': '#b98c1b'},
        {'original_name': 'dipo_diffusion_mlp_ta4_td20', 'display_name': 'DIPO', 'color': '#808080'},
        {'original_name': 'idql_diffusion_mlp_ta4_td20', 'display_name': 'IDQL', 'color': '#000000'},
        {'original_name': 'dql_diffusion_mlp_ta4_td20', 'display_name': 'DQL', 'color': '#800080'},
        {'original_name': 'rwr_diffusion_mlp_ta4_td20', 'display_name': 'DRWR', 'color': '#1E90FF'},
        {'original_name': 'awr_diffusion_mlp_ta4_td20', 'display_name': 'DAWR', 'color': '#008000'}
    ]
    
    # Update method_seed_map with display names
    updated_method_seed_map = {}
    for old_method, seed_cols in method_seed_map.items():
        for config in method_config:
            if config['original_name'] == old_method:
                updated_method_seed_map[config['display_name']] = seed_cols
                break
        else:
            updated_method_seed_map[old_method] = seed_cols
    
    print("Extracted RL Methods:")
    if updated_method_seed_map:
        for method in updated_method_seed_map:
            print(f"- {method}")
    else:
        print("None")
    
    if unmatched_columns:
        print("Unmatched columns:")
        for col in unmatched_columns:
            print(f"- {col}")
    
    # Organize data by method
    method_stats = {}
    for method, seed_cols in updated_method_seed_map.items():
        rewards = []
        for seed, col in seed_cols:
            r=data[col].dropna().values[0:n_evals]
            rewards.append(r)
        min_len=min(len(r) for r in rewards)
        truncated_rewards = [r[0:min_len] for r in rewards]
        rewards = np.array(truncated_rewards)
        
        mean_rewards = np.nanmean(rewards, axis=0)
        std_rewards = np.nanstd(rewards, axis=0)
        
        method_stats[method] = {
            'mean': mean_rewards,
            'std': std_rewards,
            'seeds': [seed for seed, _ in seed_cols]
        }
    
    # Set plot style
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {m['display_name']: m['color'] for m in method_config}
    
    handles, labels = [], []
    ours_methods = [m for m in updated_method_seed_map if '(ours)' in m]
    dppo_method = 'DPPO'
    
    for method in ours_methods:
        if method in method_stats:
            stats = method_stats[method]
            mean = stats['mean']
            std = stats['std']
            color = color_map[method]
            line, = ax.plot(x_axis[:len(mean)], mean, label=method, linewidth=3, color=color)
            ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.2, color=color)
            handles.append(line)
            labels.append(method)
    
    if dppo_method in method_stats:
        stats = method_stats[dppo_method]
        mean = stats['mean']
        std = stats['std']
        color = color_map[dppo_method]
        linewidth = 2
        alpha = 0.6
        line, = ax.plot(x_axis[:len(mean)], mean, label=dppo_method, linewidth=linewidth, color=color, alpha=alpha)
        ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.1, color=color)
        handles.append(line)
        labels.append(dppo_method)
    
    for method in updated_method_seed_map:
        if method not in ours_methods and method != dppo_method:
            stats = method_stats[method]
            mean = stats['mean']
            std = stats['std']
            color = color_map[method]
            linewidth = 2
            alpha = 0.6 if method in ['DIPO', 'IDQL', 'DQL', 'DRWR', 'DAWR', 'QSM', 'DPPO'] else 1.0
            line, = ax.plot(x_axis[:len(mean)], mean, label=method, linewidth=linewidth, color=color, alpha=alpha)
            ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.1, color=color)
            handles.append(line)
            labels.append(method)
    
    
    ax.tick_params(axis='both', labelsize=20)  # Increased tick label font size
    
    ax.set_xlabel(x_label, fontsize=22)
    if evaluation_name=='AverageEpisodeReward':
        ax.set_ylabel('Average Episode Reward', fontsize=22)
    else:
        ax.set_ylabel(evaluation_name, fontsize=22)
    # ax.legend(handles, labels)
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=16)  # Increased font size    ax.grid(True)
    if plot_x_axis == 'sample':
        # ax.set_xlim(right=7.5e7) #999*n_parallel_envs * n_rollout_steps * n_act_steps
        pass
    fig_dir = os.path.join(output_dir, environment_name, task_name)
    os.makedirs(fig_dir, exist_ok=True)
    output_file_path = os.path.join(fig_dir, output_filename)
    plt.savefig(f'{output_file_path}.png', bbox_inches='tight')
    plt.savefig(f'{output_file_path}.pdf', bbox_inches='tight')
    print(f"{evaluation_name} comparison saved to {output_file_path}.pdf")
    plt.close()
    
    return updated_method_seed_map

@hydra.main(
    version_base=None,
    config_path=os.path.join(REINFLOW_DIR, "visualize/Final_experiments/"),
    config_name="gym_config.yaml"
)
def main(cfg: OmegaConf):
    if not hasattr(cfg, 'env') or not hasattr(cfg.env, 'environment_name'):
        raise ValueError("Configuration missing 'env.environment_name'. Check gym_config.yaml.")
    
    environment_name = cfg.env.environment_name
    evaluation_name = cfg.evaluation_name
    task_name = cfg.env[environment_name].task_name
    csv_filename = cfg.env[environment_name][task_name].csv_filename
    re_expression = cfg.env[environment_name][task_name].re_expression
    output_dir = cfg.output_dir
    output_filename = cfg.output_filename
    
    plot_x_axis = cfg.plot_x_axis
    n_parallel_envs = cfg.env[environment_name][task_name].n_parallel_envs
    n_rollout_steps = cfg.env[environment_name][task_name].n_rollout_steps
    n_act_steps = cfg.env[environment_name][task_name].n_act_steps
    
    print(f"re_expression (raw)={re_expression}")
    if re_expression.startswith("r'") and re_expression.endswith("'"):
        re_expression = re_expression[2:-1]
    elif re_expression.startswith('r"') and re_expression.endswith('"'):
        re_expression = re_expression[2:-1]
    print(f"re_expression (cleaned)={re_expression}")
    
    extract_and_plot(
        evaluation_name,
        environment_name,
        task_name,
        re_expression,
        csv_filename,
        output_dir,
        output_filename,
        plot_x_axis,
        n_parallel_envs,
        n_rollout_steps,
        n_act_steps
    )

if __name__ == "__main__":
    main()