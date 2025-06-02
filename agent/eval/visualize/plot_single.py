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
import os
from util.timer import current_time
from utils import read_eval_statistics

def plot_eval_statistics(eval_statistics_,
                         inference_steps, 
                         model_name = 'ReFlow',
                         env_name = 'hopper-medium-v2',
                         label_1='Pretrained',
                         color_1='black',
                         add_denoise_step_line=True,
                         log_dir=None,
                         SAVE_FILE_NAME='compare.png',
                         plot_scale='semilogx'):
    
    # Extract metrics for Pretrained model  
    num_denoising_steps_list_, \
            avg_single_step_freq_list_, avg_single_step_freq_std_list_, \
                    avg_single_step_duration_list_, avg_single_step_duration_std_list_, \
                            avg_traj_length_list_, avg_traj_length_list_std_, \
                                avg_episode_reward_list_, avg_episode_reward_std_list_, \
                                        avg_best_reward_list_, avg_best_reward_std_list_, \
                                                success_rate_list_, success_rate_list_std_, \
                                                        num_episodes_finished_list_ = eval_statistics_            
    plot_func = plt.semilogx if plot_scale == "semilogx" else plt.plot
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot success rate
    plt.subplot(2, 3, 1)
    plot_func(num_denoising_steps_list_, success_rate_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [avg_best - std for avg_best, std in zip(success_rate_list_, success_rate_list_std_)],
                [avg_best + std for avg_best, std in zip(success_rate_list_, success_rate_list_std_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Success Rate')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()
    
    # Plot average episode reward with shading
    plt.subplot(2, 3, 2)
    plot_func(num_denoising_steps_list_, avg_episode_reward_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_, avg_episode_reward_std_list_)],
                [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_, avg_episode_reward_std_list_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Average Episode Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot inference duration (always in standard scale)
    plt.subplot(2, 3, 3)
    plt.plot(num_denoising_steps_list_, avg_single_step_duration_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [duration - std for duration, std in zip(avg_single_step_duration_list_, avg_single_step_duration_std_list_)],
                [duration + std for duration, std in zip(avg_single_step_duration_list_, avg_single_step_duration_std_list_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Inference duration')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference duration')
    plt.grid(True)
    plt.legend()
    
    
    # Plot average best reward with shading
    plt.subplot(2, 3, 4)
    plot_func(num_denoising_steps_list_, avg_best_reward_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [avg_best - std for avg_best, std in zip(avg_best_reward_list_, avg_best_reward_std_list_)],
                [avg_best + std for avg_best, std in zip(avg_best_reward_list_, avg_best_reward_std_list_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Average Best Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Best Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot average trajectory length
    plt.subplot(2, 3, 5)
    plot_func(num_denoising_steps_list_, avg_traj_length_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [avg_best - std for avg_best, std in zip(avg_traj_length_list_, avg_traj_length_list_std_)],
                [avg_best + std for avg_best, std in zip(avg_traj_length_list_, avg_traj_length_list_std_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Average Trajectory Length')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Trajectory Length')
    
    plt.grid(True)
    plt.legend()
    
    # Plot inference frequency
    plt.subplot(2, 3, 6)
    plot_func(num_denoising_steps_list_, avg_single_step_freq_list_, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_,
                [freq - std for freq, std in zip(avg_single_step_freq_list_, avg_single_step_freq_std_list_)],
                [freq + std for freq, std in zip(avg_single_step_freq_list_, avg_single_step_freq_std_list_)],
                color=color_1, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_1, linestyle='--', label=f'T={inference_steps}')
    plt.title('Inference Frequency')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference Frequency')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f"{model_name}, {env_name} \n steps = {', '.join(map(str, num_denoising_steps_list_))}", fontsize=25)
    plt.tight_layout()
    fig_path = os.path.join(log_dir, SAVE_FILE_NAME)
    plt.savefig(fig_path)
    print(f"figure saved to {fig_path}")
    plt.close()
    

if __name__ == '__main__':
    MODEL_NAME='DDIM' #'DDPM'
    FINETUNE_STEP=None #20
    LABEL1='Pretrained'
    COLOR1='black'
    ADD_DENOISE_STEP_LINE = False
    SAVE_FILE_NAME='single.png'
    PLOT_SCALE='semilogx'
    WALKER=[
            'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-14-00-00-DDPM-merged/eval_statistics.npz',
            'walker2d-v2'
            ]
    CAN=[
        'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-17-21-12-00-DDIM-can/eval_statistics.npz',
        #     'visualize/0_Models/DiffusionModel/can/25-04-16-20-33-20/eval_statistics.npz',
            'can'
    ]
    SQUARE=[
            'visualize/0_Models/DiffusionModel/square/25-04-16-20-33-18/eval_statistics.npz', 
            'square'
    ]
    _eval_path, ENV_NAME= CAN
    
    _file = read_eval_statistics(_eval_path)
    
    logdir = os.path.join('agent','eval','visualize', MODEL_NAME, 'plot_single', ENV_NAME, f'{current_time()}')
    os.makedirs(logdir, exist_ok=True)
    
    plot_eval_statistics(_file, 
                         FINETUNE_STEP, 
                         MODEL_NAME, 
                         ENV_NAME, 
                         LABEL1, 
                         COLOR1,
                         ADD_DENOISE_STEP_LINE, 
                         logdir,
                         SAVE_FILE_NAME,
                         PLOT_SCALE)