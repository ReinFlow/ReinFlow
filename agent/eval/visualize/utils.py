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


import os 
import re
from omegaconf import ListConfig
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def extract_number(label):
    """Extract the first number from a label string."""
    match = re.search(r'\d+', label)
    return int(match.group()) if match else None


def log_tick_formatter(val):
    return f"$10^{{{int(val)}}}$"


def read_eval_statistics(npz_file_path):
    loaded_data = np.load(npz_file_path)
    data = loaded_data['data']
    
    num_denoising_steps_list = data['num_denoising_steps']
    avg_single_step_freq_list = data['avg_single_step_freq']
    avg_single_step_freq_std_list = data['avg_single_step_freq_std']
    avg_single_step_duration_list = data['avg_single_step_duration']
    avg_single_step_duration_std_list = data['avg_single_step_duration_std']
    avg_traj_length_list = data['avg_traj_length']
    avg_traj_length_std_list = data['avg_traj_length_std']  # Added
    avg_episode_reward_list = data['avg_episode_reward']
    avg_best_reward_list = data['avg_best_reward']
    avg_episode_reward_std_list = data['avg_episode_reward_std']
    avg_best_reward_std_list = data['avg_best_reward_std']
    success_rate_list = data['success_rate']
    success_rate_std_list = data['success_rate_std']  # Added
    num_episodes_finished_list = data['num_episodes_finished']
    
    eval_statistics = (
        num_denoising_steps_list, 
        avg_single_step_freq_list, avg_single_step_freq_std_list,
        avg_single_step_duration_list, avg_single_step_duration_std_list,
        avg_traj_length_list, avg_traj_length_std_list,
        avg_episode_reward_list, avg_episode_reward_std_list,
        avg_best_reward_list, avg_best_reward_std_list,
        success_rate_list, success_rate_std_list,
        num_episodes_finished_list
    )
    return eval_statistics

def merge_eval_statistics(npz_file_path1, npz_file_path2, output_npz_path):
    # Read statistics from both files
    stats1 = read_eval_statistics(npz_file_path1)
    stats2 = read_eval_statistics(npz_file_path2)
    
    # Extract num_denoising_steps from both
    num_denoising_steps1 = stats1[0]
    num_denoising_steps2 = stats2[0]
    
    # Merge and sort num_denoising_steps
    merged_denoising_steps = np.concatenate([num_denoising_steps1, num_denoising_steps2])
    sort_indices = np.argsort(merged_denoising_steps)
    merged_denoising_steps = merged_denoising_steps[sort_indices]
    
    # Initialize lists to store merged data
    merged_stats = []
    
    # For each statistic, merge and sort according to sort_indices
    for i in range(len(stats1)):
        merged_data = np.concatenate([stats1[i], stats2[i]])
        merged_data = merged_data[sort_indices]
        merged_stats.append(merged_data)
    
    # Define the dtype for the structured array
    dtype = [
        ('num_denoising_steps', int),
        ('avg_single_step_freq', float),
        ('avg_single_step_freq_std', float),
        ('avg_single_step_duration', float),
        ('avg_single_step_duration_std', float),
        ('avg_traj_length', float),
        ('avg_traj_length_std', float),
        ('avg_episode_reward', float),
        ('avg_episode_reward_std', float),
        ('avg_best_reward', float),
        ('avg_best_reward_std', float),
        ('success_rate', float),
        ('success_rate_std', float),
        ('num_episodes_finished', int)
    ]
    
    # Create structured array for merged data
    data = np.zeros(len(merged_denoising_steps), dtype=dtype)
    data['num_denoising_steps'] = merged_stats[0]
    data['avg_single_step_freq'] = merged_stats[1]
    data['avg_single_step_freq_std'] = merged_stats[2]
    data['avg_single_step_duration'] = merged_stats[3]
    data['avg_single_step_duration_std'] = merged_stats[4]
    data['avg_traj_length'] = merged_stats[5]
    data['avg_traj_length_std'] = merged_stats[6]
    data['avg_episode_reward'] = merged_stats[7]
    data['avg_episode_reward_std'] = merged_stats[8]
    data['avg_best_reward'] = merged_stats[9]
    data['avg_best_reward_std'] = merged_stats[10]
    data['success_rate'] = merged_stats[11]
    data['success_rate_std'] = merged_stats[12]
    data['num_episodes_finished'] = merged_stats[13]
    
    # Save to npz file
    np.savez(output_npz_path, data=data)
    print(f"merged data saved to {output_npz_path}")
    # Return the merged statistics as a tuple
    return tuple(merged_stats)

def plot_eval_statistics(
    eval_statistics_list,
    inference_steps,
    model_name='ReFlow',
    env_name='hopper-medium-v2',
    labels=None,
    colors=None,
    add_denoise_step_line=True,
    log_dir=None,
    save_file_name='compare.png',
    plot_scale='semilogx',
    denoising_steps=None,
    legend_each=False
):
    """
    Plot evaluation statistics for multiple models with filtered denoising steps.

    Parameters:
    - eval_statistics_list: List of tuples containing evaluation statistics for each model.
    - inference_steps: Number of inference steps for vertical line.
    - model_name: Name of the model.
    - env_name: Environment name.
    - labels: List of labels for each model.
    - colors: List of colors for each model.
    - add_denoise_step_line: Whether to add a vertical line at inference_steps.
    - log_dir: Directory to save the plots.
    - save_file_name: Name of the file to save the plot.
    - plot_scale: Plot scale ('semilogx' or 'linear').
    - denoising_steps: List of denoising steps to plot (e.g., [1, 2, 4, 8, 16, 32, 64, 128, 256])
                      or tuple (min_step, max_step) for range-based filtering.
                      If None, use common steps across all datasets up to 256.
    """
    # Default labels and colors if not provided
    if labels is None:
        labels = [str(i + 1) for i in range(len(eval_statistics_list))]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(eval_statistics_list)))

    # Ensure the number of labels and colors matches the number of statistics
    if len(labels) != len(eval_statistics_list) or len(colors) != len(eval_statistics_list):
        raise ValueError("Number of labels and colors must match the number of evaluation statistics.")

    # Handle denoising_steps: convert ListConfig to list if necessary
    if isinstance(denoising_steps, ListConfig):
        denoising_steps = list(denoising_steps)

    # Determine common denoising steps
    if denoising_steps is None:
        # Find common steps up to 256
        all_steps = [set(stats[0]) for stats in eval_statistics_list]
        common_steps = sorted(set.intersection(*all_steps) & set([1, 2, 4, 8, 16, 32, 64, 128, 256]))
    elif isinstance(denoising_steps, list) and all(isinstance(s, (int, float)) for s in denoising_steps):
        common_steps = sorted(set(denoising_steps) & set.intersection(*[set(stats[0]) for stats in eval_statistics_list]))
    elif isinstance(denoising_steps, tuple) and len(denoising_steps) == 2:
        min_step, max_step = denoising_steps
        all_steps = [set(stats[0]) for stats in eval_statistics_list]
        common_steps = sorted(
            set.intersection(*all_steps) & set(s for s in all_steps[0] if min_step <= s <= max_step)
        )
    else:
        raise ValueError(
            f"denoising_steps must be a list of steps or a tuple (min_step, max_step), but got {denoising_steps}={type(denoising_steps)}"
        )

    if not common_steps:
        raise ValueError("No common denoising steps found within the specified range.")

    # Filter statistics for common steps
    filtered_stats_list = []
    for stats in eval_statistics_list:
        num_denoising_steps = np.array(stats[0])
        indices = np.isin(num_denoising_steps, common_steps)
        filtered_stats = [np.array(stat)[indices] for stat in stats]
        filtered_stats_list.append(filtered_stats)

    # Plotting function based on scale
    plot_func = plt.semilogx if plot_scale == "semilogx" else plt.plot
    plot_func2 = plt.semilogx

    # Define metric indices based on eval_statistics structure
    metric_indices = {
        'success_rate': (11, 12),  # success_rate_list, success_rate_std_list
        'avg_episode_reward': (7, 8),  # avg_episode_reward_list, avg_episode_reward_std_list
        'avg_single_step_duration': (3, 4),  # avg_single_step_duration_list, avg_single_step_duration_std_list
        'avg_best_reward': (9, 10),  # avg_best_reward_list, avg_best_reward_std_list
        'avg_traj_length': (5, 6),  # avg_traj_length_list, avg_traj_length_std_list
        'avg_single_step_freq': (1, 2)  # avg_single_step_freq_list, avg_single_step_freq_std_list
    }

    # Titles and y-labels for each subplot
    metrics = [
        ('success_rate', 'Success Rate', 'Success Rate'),
        ('avg_episode_reward', 'Average Episode Reward', 'Average Episode Reward'),
        ('avg_single_step_duration', 'Inference Duration', 'Inference Duration'),
        ('avg_best_reward', 'Average Best Reward', 'Average Best Reward'),
        ('avg_traj_length', 'Average Trajectory Length', 'Average Trajectory Length'),
        ('avg_single_step_freq', 'Inference Frequency', 'Inference Frequency')
    ]

    # Initialize figure for main plots
    plt.figure(figsize=(12, 8))

    # Plot each metric in a subplot
    for i, (metric_key, title, ylabel) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        for stats, label, color in zip(filtered_stats_list, labels, colors):
            # Extract data
            num_denoising_steps = stats[0]
            metric_value = stats[metric_indices[metric_key][0]]
            metric_std = stats[metric_indices[metric_key][1]]

            # Use linear scale for inference duration, otherwise use specified scale
            if metric_key == 'avg_single_step_duration':
                plt.plot(num_denoising_steps, metric_value, marker='o', label=label, color=color)
                plt.fill_between(
                    num_denoising_steps,
                    [val - std for val, std in zip(metric_value, metric_std)],
                    [val + std for val, std in zip(metric_value, metric_std)],
                    color=color, alpha=0.2
                )
            else:
                plot_func(num_denoising_steps, metric_value, marker='o', label=label, color=color)
                plt.fill_between(
                    num_denoising_steps,
                    [val - std for val, std in zip(metric_value, metric_std)],
                    [val + std for val, std in zip(metric_value, metric_std)],
                    color=color, alpha=0.2
                )
        
        if add_denoise_step_line and inference_steps in common_steps:
            plt.axvline(x=inference_steps, color=colors[-1], linestyle='--', label=f'T={inference_steps}')
        plt.title(title)
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel(ylabel)
        plt.grid(True)
        if legend_each: 
            plt.legend()
        elif i==6:
            plt.legend()
    # Add super title and save the figure
    plt.suptitle(
        f"{model_name}, {env_name} \n steps = {', '.join(map(str, common_steps))}", fontsize=25
    )
    plt.tight_layout()
    fig_path = os.path.join(log_dir, save_file_name)
    plt.savefig(fig_path)
    print(f"Comparison figure saved to {fig_path}")
    plt.close()

    # Plot success rate and reward vs. frequency
    plt.figure(figsize=(12, 8))

    # Success rate vs. inference frequency
    plt.subplot(1, 2, 1)
    freq_at_steps = None
    for stats, label, color in zip(filtered_stats_list, labels, colors):
        num_denoising_steps = stats[0]
        success_rate = stats[11]  # success_rate_list
        success_rate_std = stats[12]  # success_rate_std_list
        avg_single_step_freq = stats[1]  # avg_single_step_freq_list
        plot_func2(avg_single_step_freq, success_rate, marker='o', label=label, color=color)
        plt.fill_between(
            avg_single_step_freq,
            [val - std for val, std in zip(success_rate, success_rate_std)],
            [val + std for val, std in zip(success_rate, success_rate_std)],
            color=color, alpha=0.2
        )
        # Find frequency at inference_steps
        if add_denoise_step_line and freq_at_steps is None and inference_steps in common_steps:
            idx = np.where(num_denoising_steps == inference_steps)[0]
            if len(idx) > 0:
                freq_at_steps = avg_single_step_freq[idx[0]]
    if add_denoise_step_line and freq_at_steps is not None:
        plt.axvline(x=freq_at_steps, color=colors[-1], linestyle='--', label=f'Freq at T={inference_steps}')
    plt.title('Success Rate vs. Inference Frequency')
    plt.xlabel('Inference Frequency')
    plt.ylabel('Success Rate')
    if freq_at_steps is not None:
        plt.xlim(left=freq_at_steps * 0.5)
    plt.grid(True)
    plt.legend()

    # Average episode reward vs. inference frequency
    plt.subplot(1, 2, 2)
    for stats, label, color in zip(filtered_stats_list, labels, colors):
        avg_single_step_freq = stats[1]  # avg_single_step_freq_list
        avg_episode_reward = stats[7]  # avg_episode_reward_list
        avg_episode_reward_std = stats[8]  # avg_episode_reward_std_list
        plot_func2(avg_single_step_freq, avg_episode_reward, marker='o', label=label, color=color)
        plt.fill_between(
            avg_single_step_freq,
            [val - std for val, std in zip(avg_episode_reward, avg_episode_reward_std)],
            [val + std for val, std in zip(avg_episode_reward, avg_episode_reward_std)],
            color=color, alpha=0.2
        )
    if add_denoise_step_line and freq_at_steps is not None:
        plt.axvline(x=freq_at_steps, color=colors[-1], linestyle='--', label=f'Freq at T={inference_steps}')
    plt.title('Average Episode Reward vs. Inference Frequency')
    plt.xlabel('Inference Frequency')
    plt.ylabel('Average Episode Reward')
    if freq_at_steps is not None:
        plt.xlim(left=freq_at_steps * 0.5)
    plt.grid(True)
    plt.legend()

    # Save frequency comparison plot
    plt.suptitle(
        f"{model_name}, {env_name} \n Frequency Comparison \n steps = {', '.join(map(str, common_steps))}",
        fontsize=25
    )
    plt.tight_layout()
    freq_fig_path = os.path.join(log_dir, f'freq_{save_file_name}')
    plt.savefig(freq_fig_path)
    print(f"Frequency comparison figure saved to {freq_fig_path}")
    plt.close()
    

def plot_3d_eval_statistics(
    eval_statistics_list,
    inference_steps,
    model_name='ReFlow',
    env_name='hopper-medium-v2',
    labels=None,
    colors=None,
    log_dir=None,
    save_file_name='compare_3d_bar',
    plot_scale='semilogx',
    denoising_steps=None,
    legend=True,
    plot_additional_metrics=True
):
    """
    Plot 3D bar plots for evaluation statistics with (x, y, z) = (n_episodes, denoising_step, metric).

    Parameters:
    - eval_statistics_list: List of tuples containing evaluation statistics for each model.
    - inference_steps: Number of inference steps (not used but kept for consistency).
    - model_name: Name of the model.
    - env_name: Environment name.
    - labels: List of labels for each model (e.g., '8', '16', '64', ...) to extract episode numbers.
    - colors: List of colors (not used for bar plots but kept for consistency).
    - log_dir: Directory to save the plots.
    - save_file_name: Base name for saved files (not used directly).
    - plot_scale: Plot scale ('semilogx' for logarithmic x/y axes, 'linear' for linear axes).
    - denoising_steps: List of denoising steps or tuple (min_step, max_step) for filtering.
    - plot_additional_metrics: Whether to plot average episode reward and trajectory length.
    """
    # Handle denoising_steps
    if isinstance(denoising_steps, ListConfig):
        denoising_steps = list(denoising_steps)
    print(f"denoising_steps={denoising_steps}")
    
    # Determine common denoising steps
    if denoising_steps is None:
        all_steps = [set(stats[0]) for stats in eval_statistics_list]
        common_steps = sorted(set.intersection(*all_steps) & set([1, 2, 4, 8, 16, 32, 64, 128, 256]))
    elif isinstance(denoising_steps, list) and all(isinstance(s, (int, float)) for s in denoising_steps):
        common_steps = sorted(set(denoising_steps) & set.intersection(*[set(stats[0]) for stats in eval_statistics_list]))
    elif isinstance(denoising_steps, tuple) and len(denoising_steps) == 2:
        min_step, max_step = denoising_steps
        all_steps = [set(stats[0]) for stats in eval_statistics_list]
        common_steps = sorted(
            set.intersection(*all_steps) & set(s for s in all_steps[0] if min_step <= s <= max_step)
        )
    else:
        raise ValueError(
            f"denoising_steps must be a list or tuple (min_step, max_step), got {denoising_steps}"
        )
    if not common_steps:
        raise ValueError("No common denoising steps found.")
    print(f"common_steps={common_steps}")
    
    # Extract episode numbers from labels
    episode_numbers = []
    for i, label in enumerate(labels):
        num = extract_number(label)
        if num is None:
            raise ValueError(f"Could not extract episode number from label: {label}")
        episode_numbers.append(num)
    print(f"episode_numbers={episode_numbers}")
    # Create meshgrid for bar positions
    X, Y = np.meshgrid(episode_numbers, common_steps)
    
    # Filter statistics for common steps
    filtered_stats_list = []
    for stats in eval_statistics_list:
        num_denoising_steps = np.array(stats[0])
        indices = np.isin(num_denoising_steps, common_steps)
        filtered_stats = [np.array(stat)[indices] for stat in stats]
        filtered_stats_list.append(filtered_stats)
    # Define metrics to plot
    metrics_to_plot = [
        ('success_rate', 11, 'Success Rate')
    ]
    if plot_additional_metrics:
        metrics_to_plot.extend([
            ('avg_episode_reward', 7, 'Average Episode Reward'),
            ('avg_traj_length', 5, 'Average Trajectory Length')
        ])

    # Create 3D bar plots for each metric
    for metric_key, index, title in metrics_to_plot:
        # Create Z data (metric values)
        Z = np.zeros((len(common_steps), len(episode_numbers)))
        for i, stats in enumerate(filtered_stats_list):
            Z[:, i] = stats[index]  # Fill column-wise for each episode number
        # Flatten arrays for bar plotting
        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = Z.ravel()

        # Log-transform x and y data
        x_log = np.log10(x_flat)
        y_log = np.log10(y_flat)

        # Create grid for interpolation in log space
        x_unique_log = np.unique(x_log)
        y_unique_log = np.unique(y_log)
        X_log, Y_log = np.meshgrid(x_unique_log, y_unique_log)

        # Interpolate z values onto the log grid
        points_log = np.vstack((x_log, y_log)).T
        Z = griddata(points_log, z_flat, (X_log, Y_log), method='cubic')

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X_log, Y_log, Z, cmap='Purples', edgecolor='none')
        print(f"$$$ Z={Z}")
        ax.set_xticks(x_unique_log)
        ax.set_xticklabels([log_tick_formatter(v) for v in x_unique_log])
        ax.set_yticks(y_unique_log)
        ax.set_yticklabels([log_tick_formatter(v) for v in y_unique_log])

        ax.set_xlabel('Episodes Number')
        ax.set_ylabel('Inference Step')
        ax.set_zlabel('metric_key')
        ax.set_title(f'{metric_key} of {model_name} in {env_name}\n Inference Steps={common_steps}\n Episode Numbers={episode_numbers}')

        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.view_init(elev=20, azim=30)
        # Save figure as PNG and PDF
        fig_path_png = os.path.join(log_dir, f"{metric_key}_3d_surface.png")
        fig_path_pdf = os.path.join(log_dir, f"{metric_key}_3d_surface.pdf")
        plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"3D bar plot for {title} saved to {fig_path_png} and {fig_path_pdf}")
        plt.close()
        
def plot_3d_eval_statistics_groups(
    eval_statistics_groups,
    group_labels,
    inference_steps,
    model_name='ReFlow',
    env_name='hopper-medium-v2',
    labels=None,
    colors=None,
    log_dir=None,
    save_file_name='compare_3d_surface',
    plot_scale='semilogx',
    denoising_steps=None,
    legend=True,
    plot_additional_metrics=True,
    finetuned_cmap='Purples',
    pretrained_cmap='PuRd'
):
    """
    Plot multiple 3D surface plots in a single figure to compare evaluation statistics across different groups,
    such as finetuned and pretrained models. The axes represent (x, y, z) = (number of pretrained episodes,
    denoising steps, metric value), with logarithmic scaling on x and y axes by default. The finetuned surface
    uses the mean across three seeds, with shading (via alpha transparency) indicating the standard deviation
    (std) across seeds: higher std results in lighter, more transparent areas. Both surfaces use purple tones:
    'Purples' for a darker purple finetuned surface and 'PuRd' for a lighter purple (light velvet) pretrained
    surface. A legend distinguishes the groups. The function supports plotting multiple metrics (success rate,
    average episode reward, average trajectory length) and saves the plots as PNG and PDF files.

    Parameters:
    - eval_statistics_groups: List of lists, where each sublist contains evaluation statistics. For finetuned group,
      each entry is a tuple of (mean_stats, std_stats), where mean_stats and std_stats are tuples of
      (denoising_steps, metric1, metric2, ...). For pretrained group, each entry is a tuple of
      (denoising_steps, metric1, metric2, ...).
    - group_labels: List of labels for each group (e.g., ['Finetuned', 'Pretrained']).
    - inference_steps: Number of inference steps (included for compatibility, not used in plotting).
    - model_name: Name of the model (default: 'ReFlow').
    - env_name: Name of the environment (default: 'hopper-medium-v2').
    - labels: List of episode number labels (e.g., ['8', '16', '32', ...]).
    - colors: List of colors (included for compatibility, not used in surface plots).
    - log_dir: Directory to save the output plot files.
    - save_file_name: Base name for saved plot files (default: 'compare_3d_surface').
    - plot_scale: Scale for axes ('semilogx' for logarithmic x/y, 'linear' for linear; default: 'semilogx').
    - denoising_steps: List of denoising steps or tuple (min_step, max_step) to filter steps (default: None).
    - legend: Whether to include a legend in the plot (default: True).
    - plot_additional_metrics: Whether to plot additional metrics like average episode reward and trajectory length
      (default: True).
    - finetuned_cmap: Matplotlib colormap name for the finetuned surface (default: 'Purples' for darker purple).
    - pretrained_cmap: Matplotlib colormap name for the pretrained surface (default: 'PuRd' for lighter purple).
    """
    # Handle denoising_steps
    if isinstance(denoising_steps, ListConfig):
        denoising_steps = list(denoising_steps)
    print(f"denoising_steps={denoising_steps}")
    
    # Determine common denoising steps across all groups
    all_steps = []
    for group in eval_statistics_groups:
        for stats in group:
            # Handle finetuned (mean, std) tuples vs pretrained stats
            stats_data = stats[0] if isinstance(stats, tuple) and len(stats) == 2 else stats
            all_steps.append(set(stats_data[0]))
    if denoising_steps is None:
        common_steps = sorted(set.intersection(*all_steps) & set([1, 2, 4, 8, 16, 32, 64, 128, 256]))
    elif isinstance(denoising_steps, list) and all(isinstance(s, (int, float)) for s in denoising_steps):
        common_steps = sorted(set(denoising_steps) & set.intersection(*all_steps))
    elif isinstance(denoising_steps, tuple) and len(denoising_steps) == 2:
        min_step, max_step = denoising_steps
        common_steps = sorted(
            set.intersection(*all_steps) & set(s for s in all_steps[0] if min_step <= s <= max_step)
        )
    else:
        raise ValueError(
            f"denoising_steps must be a list or tuple (min_step, max_step), got {denoising_steps}"
        )
    if not common_steps:
        raise ValueError("No common denoising steps found.")
    print(f"common_steps={common_steps}")
    
    # Extract episode numbers from labels
    episode_numbers = []
    for label in labels:
        num = extract_number(label)
        if num is None:
            raise ValueError(f"Could not extract episode number from label: {label}")
        episode_numbers.append(num)
    print(f"episode_numbers={episode_numbers}")
    
    # Define metrics to plot
    metrics_to_plot = [('success_rate', 11, 'Success Rate')]
    if plot_additional_metrics:
        metrics_to_plot.extend([
            ('avg_episode_reward', 7, 'Average Episode Reward'),
            ('avg_traj_length', 5, 'Average Trajectory Length')
        ])

    # Create 3D surface plots for each metric
    for metric_key, index, title in metrics_to_plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmaps = [finetuned_cmap, pretrained_cmap]  # Purple-based color maps
        
        for group_idx, eval_statistics_list in enumerate(eval_statistics_groups):
            # Filter statistics for common steps
            filtered_stats_list = []
            for stats in eval_statistics_list:
                if group_idx == 0:  # Finetuned: stats = (mean_stats, std_stats)
                    mean_stats, std_stats = stats
                    num_denoising_steps = np.array(mean_stats[0])
                    indices = np.isin(num_denoising_steps, common_steps)
                    filtered_mean_stats = [np.array(stat)[indices] for stat in mean_stats]
                    filtered_std_stats = [np.array(stat)[indices] for stat in std_stats]
                    filtered_stats_list.append((filtered_mean_stats, filtered_std_stats))
                else:  # Pretrained: stats = (denoising_steps, metrics...)
                    num_denoising_steps = np.array(stats[0])
                    indices = np.isin(num_denoising_steps, common_steps)
                    filtered_stats = [np.array(stat)[indices] for stat in stats]
                    filtered_stats_list.append(filtered_stats)
            
            # Create Z data (metric values) and std data (for finetuned)
            Z = np.zeros((len(common_steps), len(episode_numbers)))
            if group_idx == 0:  # Finetuned
                Z_std = np.zeros((len(common_steps), len(episode_numbers)))
            
            for i, stats in enumerate(filtered_stats_list):
                if group_idx == 0:  # Finetuned
                    mean_stats, std_stats = stats
                    Z[:, i] = mean_stats[index]
                    Z_std[:, i] = std_stats[index]
                else:  # Pretrained
                    Z[:, i] = stats[index]
            
            # Create meshgrid for plotting
            X, Y = np.meshgrid(episode_numbers, common_steps)
            x_flat = X.ravel()
            y_flat = Y.ravel()
            z_flat = Z.ravel()
            
            # Log-transform x and y data
            x_log = np.log10(x_flat)
            y_log = np.log10(y_flat)
            
            # Create grid for interpolation in log space
            x_unique_log = np.unique(x_log)
            y_unique_log = np.unique(y_log)
            X_log, Y_log = np.meshgrid(x_unique_log, y_unique_log)
            
            # Interpolate z values
            points_log = np.vstack((x_log, y_log)).T
            Z_interp = griddata(points_log, z_flat, (X_log, Y_log), method='cubic')
            
            # Plot surface
            cmap = cmaps[group_idx]
            if group_idx == 0:  # Finetuned: Apply std-based shading
                # Interpolate std values
                Z_std_flat = Z_std.ravel()
                Z_std_interp = griddata(points_log, Z_std_flat, (X_log, Y_log), method='cubic')
                # Normalize std for alpha (0.3 to 0.9 range for visibility)
                std_min, std_max = np.nanmin(Z_std_interp), np.nanmax(Z_std_interp)
                if std_max > std_min:
                    alpha = 0.3 + 0.6 * (1 - (Z_std_interp - std_min) / (std_max - std_min))
                else:
                    alpha = np.full_like(Z_std_interp, 0.7)  # Default alpha if std is constant
                # Compute face colors with std-based alpha
                norm_Z = (Z_interp - np.nanmin(Z_interp)) / (np.nanmax(Z_interp) - np.nanmin(Z_interp))
                colors = plt.get_cmap(cmap)(norm_Z)
                colors[..., 3] = np.clip(alpha, 0.3, 0.9)  # Apply alpha to face colors
                surf = ax.plot_surface(X_log, Y_log, Z_interp, facecolors=colors, edgecolor='none')
            else:  # Pretrained
                surf = ax.plot_surface(X_log, Y_log, Z_interp, cmap=cmap, edgecolor='none', alpha=0.7)
            
            # Add dummy line for legend
            representative_color = plt.get_cmap(cmap)(0.5)
            ax.plot([], [], color=representative_color, label=group_labels[group_idx])
        
        # Customize axes
        ax.set_xticks(x_unique_log)
        ax.set_xticklabels([log_tick_formatter(v) for v in x_unique_log])
        ax.set_yticks(y_unique_log)
        ax.set_yticklabels([log_tick_formatter(v) for v in y_unique_log])
        
        ax.set_xlabel('Pretrained Episodes') # (log scale)
        ax.set_ylabel('Inference Steps') #(log scale)
        ax.set_zlabel(metric_key)
        ax.set_title(f'{title} of {model_name} in {env_name}')
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.view_init(elev=20, azim=30)
        
        if legend:
            ax.legend()
        
        # Save figure
        fig_path_png = os.path.join(log_dir, f"{metric_key}_3d_surface.png")
        fig_path_pdf = os.path.join(log_dir, f"{metric_key}_3d_surface.pdf")
        plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"3D surface plot for {title} saved to {fig_path_png} and {fig_path_pdf}")
        plt.close()





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os

def plot_3d_eval_statistics_groups_same_color(
    eval_statistics_groups,
    group_labels,
    inference_steps,
    model_name='ReFlow',
    env_name='hopper-medium-v2',
    labels=None,
    colors=None,
    log_dir=None,
    save_file_name='compare_3d_surface',
    plot_scale='semilogx',
    denoising_steps=None,
    legend=True,
    plot_additional_metrics=True,
    finetuned_color='purple',  # Single color for finetuned surface (darker purple)
    pretrained_color='plum'    # Single color for pretrained surface (lighter purple)
):
    """
    Plot multiple 3D surface plots in a single figure to compare evaluation statistics across different groups,
    such as finetuned and pretrained models. The axes represent (x, y, z) = (number of pretrained episodes,
    denoising steps, metric value), with logarithmic scaling on x and y axes by default. The finetuned surface
    uses a single color (darker purple) with shading via alpha transparency indicating the standard deviation
    (std) across seeds: higher std results in lighter, more transparent areas. The pretrained surface uses a
    single lighter purple color with constant alpha. A legend distinguishes the groups. The function supports
    plotting multiple metrics (success rate, average episode reward, average trajectory length) and saves the
    plots as PNG and PDF files. The z-axis label is placed on top of the z-axis to avoid clipping and improve visibility.

    Parameters:
    - eval_statistics_groups: List of lists, where each sublist contains evaluation statistics. For finetuned group,
      each entry is a tuple of (mean_stats, std_stats), where mean_stats and std_stats are tuples of
      (denoising_steps, metric1, metric2, ...). For pretrained group, each entry is a tuple of
      (denoising_steps, metric1, metric2, ...).
    - group_labels: List of labels for each group (e.g., ['Finetuned', 'Pretrained']).
    - inference_steps: Number of inference steps (included for compatibility, not used in plotting).
    - model_name: Name of the model (default: 'ReFlow').
    - env_name: Name of the environment (default: 'hopper-medium-v2').
    - labels: List of episode number labels (e.g., ['8', '16', '32', ...]).
    - colors: List of colors (included for compatibility, not used in surface plots).
    - log_dir: Directory to save the output plot files.
    - save_file_name: Base name for saved plot files (default: 'compare_3d_surface').
    - plot_scale: Scale for axes ('semilogx' for logarithmic x/y, 'linear' for linear; default: 'semilogx').
    - denoising_steps: List of denoising steps or tuple (min_step, max_step) to filter steps (default: None).
    - legend: Whether to include a legend in the plot (default: True).
    - plot_additional_metrics: Whether to plot additional metrics like average episode reward and trajectory length
      (default: True).
    - finetuned_color: Color for the finetuned surface (default: 'purple').
    - pretrained_color: Color for the pretrained surface (default: 'plum').
    """
    # Handle denoising_steps
    if isinstance(denoising_steps, ListConfig):
        denoising_steps = list(denoising_steps)
    print(f"denoising_steps={denoising_steps}")
    
    # Determine common denoising steps across all groups
    all_steps = []
    for group in eval_statistics_groups:
        for stats in group:
            # Handle finetuned (mean, std) tuples vs pretrained stats
            stats_data = stats[0] if isinstance(stats, tuple) and len(stats) == 2 else stats
            all_steps.append(set(stats_data[0]))
    if denoising_steps is None:
        common_steps = sorted(set.intersection(*all_steps) & set([1, 2, 4, 8, 16, 32, 64, 128, 256]))
    elif isinstance(denoising_steps, list) and all(isinstance(s, (int, float)) for s in denoising_steps):
        common_steps = sorted(set(denoising_steps) & set.intersection(*all_steps))
    elif isinstance(denoising_steps, tuple) and len(denoising_steps) == 2:
        min_step, max_step = denoising_steps
        common_steps = sorted(
            set.intersection(*all_steps) & set(s for s in all_steps[0] if min_step <= s <= max_step)
        )
    else:
        raise ValueError(
            f"denoising_steps must be a list or tuple (min_step, max_step), got {denoising_steps}"
        )
    if not common_steps:
        raise ValueError("No common denoising steps found.")
    print(f"common_steps={common_steps}")
    
    # Extract episode numbers from labels
    episode_numbers = []
    for label in labels:
        num = extract_number(label)
        if num is None:
            raise ValueError(f"Could not extract episode number from label: {label}")
        episode_numbers.append(num)
    print(f"episode_numbers={episode_numbers}")
    
    # Define metrics to plot
    metrics_to_plot = [('success_rate', 11, 'Success Rate')]
    if plot_additional_metrics:
        metrics_to_plot.extend([
            ('avg_episode_reward', 7, 'Average Episode Reward'),
            ('avg_traj_length', 5, 'Average Trajectory Length')
        ])

    # Define fixed colors for surfaces
    surface_colors = [finetuned_color, pretrained_color]  # ['purple', 'plum']

    # Create 3D surface plots for each metric
    for metric_key, index, title in metrics_to_plot:
        fig = plt.figure(figsize=(14, 10))  # Increased figure size for more space
        ax = fig.add_subplot(111, projection='3d')
        
        for group_idx, eval_statistics_list in enumerate(eval_statistics_groups):
            # Filter statistics for common steps
            filtered_stats_list = []
            for stats in eval_statistics_list:
                if group_idx == 0:  # Finetuned: stats = (mean_stats, std_stats)
                    mean_stats, std_stats = stats
                    num_denoising_steps = np.array(mean_stats[0])
                    indices = np.isin(num_denoising_steps, common_steps)
                    filtered_mean_stats = [np.array(stat)[indices] for stat in mean_stats]
                    filtered_std_stats = [np.array(stat)[indices] for stat in std_stats]
                    filtered_stats_list.append((filtered_mean_stats, filtered_std_stats))
                else:  # Pretrained: stats = (denoising_steps, metrics...)
                    num_denoising_steps = np.array(stats[0])
                    indices = np.isin(num_denoising_steps, common_steps)
                    filtered_stats = [np.array(stat)[indices] for stat in stats]
                    filtered_stats_list.append(filtered_stats)
            
            # Create Z data (metric values) and std data (for finetuned)
            Z = np.zeros((len(common_steps), len(episode_numbers)))
            if group_idx == 0:  # Finetuned
                Z_std = np.zeros((len(common_steps), len(episode_numbers)))
            
            for i, stats in enumerate(filtered_stats_list):
                if group_idx == 0:  # Finetuned
                    mean_stats, std_stats = stats
                    Z[:, i] = mean_stats[index]
                    Z_std[:, i] = std_stats[index]
                else:  # Pretrained
                    Z[:, i] = stats[index]
            
            # Create meshgrid for plotting
            X, Y = np.meshgrid(episode_numbers, common_steps)
            x_flat = X.ravel()
            y_flat = Y.ravel()
            z_flat = Z.ravel()
            
            # Log-transform x and y data
            x_log = np.log10(x_flat)
            y_log = np.log10(y_flat)
            
            # Create grid for interpolation in log space
            x_unique_log = np.unique(x_log)
            y_unique_log = np.unique(y_log)
            X_log, Y_log = np.meshgrid(x_unique_log, y_unique_log)
            
            # Interpolate z values
            points_log = np.vstack((x_log, y_log)).T
            Z_interp = griddata(points_log, z_flat, (X_log, Y_log), method='cubic')
            
            # Plot surface with single color
            if group_idx == 0:  # Finetuned: Apply std-based shading
                # Interpolate std values
                Z_std_flat = Z_std.ravel()
                Z_std_interp = griddata(points_log, Z_std_flat, (X_log, Y_log), method='cubic')
                # Normalize std for alpha (0.3 to 0.9 range for visibility)
                std_min, std_max = np.nanmin(Z_std_interp), np.nanmax(Z_std_interp)
                if std_max > std_min:
                    alpha = 0.3 + 0.6 * (1 - (Z_std_interp - std_min) / (std_max - std_min))
                else:
                    alpha = np.full_like(Z_std_interp, 0.7)  # Default alpha if std is constant
                # Use single color with std-based alpha
                colors = np.full((*Z_interp.shape, 4), plt.get_cmap('viridis')(0.0))  # Temporary colormap to get RGBA
                colors[..., :3] = plt.matplotlib.colors.to_rgb(finetuned_color)  # Set RGB to single color
                colors[..., 3] = np.clip(alpha, 0.3, 0.9)  # Apply alpha
                surf = ax.plot_surface(X_log, Y_log, Z_interp, facecolors=colors, edgecolor='none')
            else:  # Pretrained: Use single color with constant alpha
                colors = np.full((*Z_interp.shape, 4), plt.get_cmap('viridis')(0.0))  # Temporary colormap to get RGBA
                colors[..., :3] = plt.matplotlib.colors.to_rgb(pretrained_color)  # Set RGB to single color
                colors[..., 3] = 0.7  # Constant alpha
                surf = ax.plot_surface(X_log, Y_log, Z_interp, facecolors=colors, edgecolor='none')
            
            # Add dummy line for legend
            ax.plot([], [], color=surface_colors[group_idx], label=group_labels[group_idx])
        
        # Customize axes
        ax.set_xticks(x_unique_log)
        ax.set_xticklabels([log_tick_formatter(v) for v in x_unique_log], fontsize=12)
        ax.set_xlabel('Pre-trained Episode', fontsize=18, labelpad=10)
        
        
        ax.set_yticks(y_unique_log)
        ax.set_yticklabels([log_tick_formatter(v) for v in y_unique_log], fontsize=12)
        ax.set_ylabel('Inference Step', fontsize=20, labelpad=10)
        
        ax.set_zticklabels(ax.get_zticks(), fontsize=16)  # Smaller z-axis tick labels
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Add this line to show tickles in int.
        
        
        # Place z-axis label on top using ax.text
        if 'avg_episode_reward' == metric_key:
            z_label = 'Average Episode Reward'
        elif 'rate' in metric_key:
            z_label = 'Success Rate'
        else:
            z_label = title
        # Get axis limits to position label at the top of z-axis
        z_min, z_max = ax.get_zlim()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Place label above the z-axis in data coordinates
        # Position: center of x-axis, max of y-axis, slightly above z_max
        ax.text(
            x=(0.6*x_min + 0.4*x_max),  # Center of x-axis
            y=y_max,                # Max of y-axis (top of plot)
            z=z_max + (z_max - z_min) * 0.16,  # Slightly above z_max
            s=z_label,
            fontsize=18,
            ha='center',
            va='bottom',
            transform=ax.transData  # Use 3D data coordinates
        )
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.view_init(elev=20, azim=30)
        # if legend:
        #     ax.legend(fontsize=18) #, loc='upper left', bbox_to_anchor=(0.0, 1.0)
        
        # Adjust layout to prevent clipping
        fig.subplots_adjust(left=0.1, right=0.8, top=0.85, bottom=0.15)  # Adjusted margins
        
        # Save figure
        fig_path_png = os.path.join(log_dir, f"{metric_key}_3d_surface.png")
        fig_path_pdf = os.path.join(log_dir, f"{metric_key}_3d_surface.pdf")
        plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"\n\n 3D surface plot for {title} saved to \n{fig_path_png}\n{fig_path_pdf}\n\n")
        plt.show()
        plt.close()