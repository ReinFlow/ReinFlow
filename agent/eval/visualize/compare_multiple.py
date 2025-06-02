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


import matplotlib.pyplot as plt
import numpy as np
from util.timer import current_time
from utils import read_eval_statistics
import os 

def plot_eval_statistics(eval_statistics_list,
                         inference_steps,
                         model_name='ReFlow',
                         env_name='hopper-medium-v2',
                         labels=None,
                         colors=None,
                         add_denoise_step_line=True,
                         log_dir=None,
                         save_file_name='compare.png',
                         plot_scale='semilogx'):
    """
    Plot evaluation statistics for multiple models.

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
    """
    # Default labels and colors if not provided
    if labels is None:
        labels = [str(i+1) for i in range(len(eval_statistics_list))]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(eval_statistics_list)))

    # Ensure the number of labels and colors matches the number of statistics
    if len(labels) != len(eval_statistics_list) or len(colors) != len(eval_statistics_list):
        raise ValueError("Number of labels and colors must match the number of evaluation statistics.")

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
        for stats, label, color in zip(eval_statistics_list, labels, colors):
            # Extract data
            num_denoising_steps = stats[0]
            metric_value = stats[metric_indices[metric_key][0]]
            metric_std = stats[metric_indices[metric_key][1]]

            # Use linear scale for inference duration, otherwise use specified scale
            if metric_key == 'avg_single_step_duration':
                plt.plot(num_denoising_steps, metric_value, marker='o', label=label, color=color)
                plt.fill_between(num_denoising_steps,
                                 [val - std for val, std in zip(metric_value, metric_std)],
                                 [val + std for val, std in zip(metric_value, metric_std)],
                                 color=color, alpha=0.2)
            else:
                plot_func(num_denoising_steps, metric_value, marker='o', label=label, color=color)
                plt.fill_between(num_denoising_steps,
                                 [val - std for val, std in zip(metric_value, metric_std)],
                                 [val + std for val, std in zip(metric_value, metric_std)],
                                 color=color, alpha=0.2)

        if add_denoise_step_line:
            plt.axvline(x=inference_steps, color=colors[-1], linestyle='--', label=f'T={inference_steps}')
        plt.title(title)
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

    # Add super title and save the figure
    plt.suptitle(f"{model_name}, {env_name} \n steps = {', '.join(map(str, eval_statistics_list[0][0]))}", fontsize=25)
    plt.tight_layout()
    import os
    fig_path = os.path.join(log_dir,save_file_name)
    plt.savefig(fig_path)
    print(f"Comparison figure saved to {fig_path}")
    plt.close()

    # Plot success rate and reward vs. frequency
    plt.figure(figsize=(12, 8))

    # Success rate vs. inference frequency
    plt.subplot(1, 2, 1)
    freq_at_steps = None
    for stats, label, color in zip(eval_statistics_list, labels, colors):
        num_denoising_steps = stats[0]
        success_rate = stats[11]  # success_rate_list
        success_rate_std = stats[12]  # success_rate_std_list
        avg_single_step_freq = stats[1]  # avg_single_step_freq_list
        plot_func2(avg_single_step_freq, success_rate, marker='o', label=label, color=color)
        plt.fill_between(avg_single_step_freq,
                         [val - std for val, std in zip(success_rate, success_rate_std)],
                         [val + std for val, std in zip(success_rate, success_rate_std)],
                         color=color, alpha=0.2)
        # Find frequency at inference_steps
        if add_denoise_step_line and freq_at_steps is None:
            idx = np.where(num_denoising_steps == inference_steps)[0]
            if len(idx) > 0:
                freq_at_steps = avg_single_step_freq[idx[0]]
    if add_denoise_step_line and freq_at_steps is not None:
        plt.axvline(x=freq_at_steps, color=colors[-1], linestyle='--', label=f'Freq at T={inference_steps}')
    plt.title('Success Rate vs. Inference Frequency')
    plt.xlabel('Inference Frequency')
    plt.ylabel('Success Rate')
    if freq_at_steps is not None: plt.xlim(left=freq_at_steps*0.5)
    plt.grid(True)
    plt.legend()
    
    # Average episode reward vs. inference frequency
    plt.subplot(1, 2, 2)
    for stats, label, color in zip(eval_statistics_list, labels, colors):
        avg_single_step_freq = stats[1]  # avg_single_step_freq_list
        avg_episode_reward = stats[7]  # avg_episode_reward_list
        avg_episode_reward_std = stats[8]  # avg_episode_reward_std_list
        plot_func2(avg_single_step_freq, avg_episode_reward, marker='o', label=label, color=color)
        plt.fill_between(avg_single_step_freq,
                         [val - std for val, std in zip(avg_episode_reward, avg_episode_reward_std)],
                         [val + std for val, std in zip(avg_episode_reward, avg_episode_reward_std)],
                         color=color, alpha=0.2)
    if add_denoise_step_line and freq_at_steps is not None:
        plt.axvline(x=freq_at_steps, color=colors[-1], linestyle='--', label=f'Freq at T={inference_steps}')
    plt.title('Average Episode Reward vs. Inference Frequency')
    plt.xlabel('Inference Frequency')
    plt.ylabel('Average Episode Reward')
    if freq_at_steps is not None: plt.xlim(left=freq_at_steps*0.5)
    plt.grid(True)
    plt.legend()
    
    # Save frequency comparison plot
    plt.suptitle(f"{model_name}, {env_name} \n Frequency Comparison \n steps = {', '.join(map(str, eval_statistics_list[0][0]))}", fontsize=25)
    plt.tight_layout()
    freq_fig_path = os.path.join(log_dir,f'freq_{save_file_name}')
    plt.savefig(freq_fig_path)
    plt.close()




import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reward_vs_denoising(eval_statistics_list,
                             inference_steps,
                             model_name='ReFlow',
                             env_name='hopper-medium-v2',
                             labels=None,
                             colors=None,
                             add_denoise_step_line=True,
                             log_dir=None,
                             save_file_name='only_reward.png',
                             plot_scale='semilogx'):
    """
    Plot average episode reward vs. denoising steps for multiple models.

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
    """
    # Default labels and colors if not provided
    if labels is None:
        labels = [str(i+1) for i in range(len(eval_statistics_list))]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(eval_statistics_list)))

    # Ensure the number of labels and colors matches the number of statistics
    if len(labels) != len(eval_statistics_list) or len(colors) != len(eval_statistics_list):
        raise ValueError("Number of labels and colors must match the number of evaluation statistics.")

    # Plotting function based on scale
    plot_func = plt.semilogx if plot_scale == "semilogx" else plt.plot

    # Initialize figure
    plt.figure(figsize=(13, 9))

    # Plot average episode reward
    for stats, label, color in zip(eval_statistics_list, labels, colors):
        num_denoising_steps = stats[0]
        avg_episode_reward = stats[7]  # avg_episode_reward_list
        
        # Filter data to include only indices where num_denoising_steps <= inference_steps
        mask = num_denoising_steps <= 64
        filtered_steps = num_denoising_steps[mask]
        filtered_rewards = avg_episode_reward[mask]

        plot_func(filtered_steps, filtered_rewards, marker='o', label=label, color=color, linewidth=8, markersize=16)  # Thicker lines

    if add_denoise_step_line:
        plt.axvline(x=inference_steps, color='black', linestyle='--', linewidth=8)  # Thicker vertical line

    plt.xlabel('Number of Denoising Steps', fontsize=24)
    plt.ylabel('Average Episode Reward', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)  # Enlarge tick labels
    plt.grid(True)
    # Enlarge legend panel with larger font and thicker lines
    plt.legend(fontsize=24, handlelength=2, handletextpad=0.5, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels), frameon=True, prop={'size': 24})
    
    # Add super title
    title_text = f"{model_name}, {env_name}\nsteps = {', '.join(map(str, eval_statistics_list[0][0][eval_statistics_list[0][0] <= inference_steps]))}"
    print(f"title_text={title_text}")
    # Save the figure as PNG and PDF
    plt.tight_layout()
    if log_dir is not None:
        fig_path_png = os.path.join(log_dir, save_file_name)
        fig_path_pdf = os.path.join(log_dir, save_file_name.replace('.png', '.pdf'))
        plt.savefig(fig_path_png, format='png', bbox_inches='tight')
        plt.savefig(fig_path_pdf, format='pdf', bbox_inches='tight')
        print(f"Reward figure saved to {fig_path_png} and {fig_path_pdf}")
    else:
        plt.savefig(save_file_name, format='png', bbox_inches='tight')
        plt.savefig(save_file_name.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Reward figure saved to {save_file_name} and {save_file_name.replace('.png', '.pdf')}")

    plt.close()
    

if __name__ == '__main__':
    MODEL_NAME = 'Denoising Models'
    PRETRAIN_STEP =20 # 100
    LABELS = ['1-ReFlow', 'Shortcut', 'DDPM', 'DDIM'] #['1-ReFlow', 'DDIM']   #['1-ReFlow', 'DDPM', 'DDIM']
    COLORS = ['purple', 'red', 'orange', 'brown']  #['purple', 'red']      #['purple', 'orange', 'red']
    ADD_DENOISE_STEP_LINE = True
    SAVE_FILE_NAME = 'compare.png'
    PLOT_SCALE = 'semilogx'
    COMPARE_NAME='DDIM_Flow' #'DDPM_DDIM_Flow'
    WALKER = [
        (
        'visualize/0_Models/ReFlow/walker2d-medium-v2/25-04-16-15-27-42/eval_statistics.npz',
        'visualize/0_Models/ShortCutFlow/walker2d-medium-v2/25-04-25-23-29-01/eval_statistics.npz',
        'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-15-10-50/eval_statistics.npz',
        'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-15-11-56/eval_statistics.npz',
        ),
        'walker2d-medium-v2'
    ]
    
    CAN=[
        ('visualize/0_Models/ReFlow/can/25-04-16-21-54-53/eval_statistics.npz',
        'visualize/0_Models/DiffusionModel/can/25-04-17-11-13-29/eval_statistics.npz') # DDIM)
        ,
        'can'
    ]
    
    SQUARE=[
        (
            'visualize/0_Models/ReFlow/square/25-04-17-10-04-06/eval_statistics.npz',
            'visualize/0_Models/DiffusionModel/square/25-04-17-11-13-30/eval_statistics.npz',
            ),
        'square'
    ]
    
    
    
    eval_paths, ENV_NAME = WALKER
    # Read evaluation statistics
    eval_stats = [read_eval_statistics(eval_path) for eval_path in eval_paths]
    # Create log directory
    logdir = f"visualize/{MODEL_NAME}/{COMPARE_NAME}/{ENV_NAME}/{current_time()}"
    os.makedirs(logdir, exist_ok=True)
    
    
    
    plot_reward_vs_denoising(
        eval_statistics_list=eval_stats,
        inference_steps=PRETRAIN_STEP,
        model_name=MODEL_NAME,
        env_name=ENV_NAME,
        labels=LABELS,
        colors=COLORS,
        add_denoise_step_line=ADD_DENOISE_STEP_LINE,
        log_dir=logdir,
        save_file_name=SAVE_FILE_NAME,
    )
    exit()
    
    
    
    
    # Call the plotting function
    plot_eval_statistics(
        eval_statistics_list=eval_stats,
        inference_steps=PRETRAIN_STEP,
        model_name=MODEL_NAME,
        env_name=ENV_NAME,
        labels=LABELS,
        colors=COLORS,
        add_denoise_step_line=ADD_DENOISE_STEP_LINE,
        log_dir=logdir,
        save_file_name=SAVE_FILE_NAME,
        plot_scale=PLOT_SCALE
    )