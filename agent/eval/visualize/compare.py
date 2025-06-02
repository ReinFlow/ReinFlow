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

def plot_eval_statistics(eval_statistics_pretrain,
                         eval_statistics_finetune,
                         inference_steps, 
                         model_name = 'ReFlow',
                         env_name = 'hopper-medium-v2',
                         label_1='Pretrained',
                         color_1='black',
                         label_2='Finetuned',
                         color_2='red',
                         add_denoise_step_line=True,
                         log_dir=None,
                         SAVE_FILE_NAME='compare.png',
                         plot_scale='semilogx'):
    
    # Extract metrics for Pretrained model  
    num_denoising_steps_list_pretrain, \
            avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain, \
                    avg_single_step_duration_list_pretrain, avg_single_step_duration_std_list_pretrain, \
                            avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain, \
                                avg_episode_reward_list_pretrain, avg_episode_reward_std_list_pretrain, \
                                        avg_best_reward_list_pretrain, avg_best_reward_std_list_pretrain, \
                                                success_rate_list_pretrain, success_rate_list_std_pretrain, \
                                                        num_episodes_finished_list_pretrain = eval_statistics_pretrain        
    # Extract metrics for Finetuned model
    num_denoising_steps_list_finetune, \
            avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune, \
                    avg_single_step_duration_list_finetune, avg_single_step_duration_std_list_finetune, \
                            avg_traj_length_list_finetune, avg_traj_length_list_std_finetune, \
                                avg_episode_reward_list_finetune, avg_episode_reward_std_list_finetune, \
                                        avg_best_reward_list_finetune, avg_best_reward_std_list_finetune, \
                                                success_rate_list_finetune, success_rate_list_std_finetune, \
                                                        num_episodes_finished_list_finetune = eval_statistics_finetune
    
    plot_func = plt.semilogx if plot_scale == "semilogx" else plt.plot
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot average episode reward with shading
    plt.subplot(2, 3, 1)
    plot_func(num_denoising_steps_list_pretrain, avg_episode_reward_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_pretrain, avg_episode_reward_std_list_pretrain)],
                [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_pretrain, avg_episode_reward_std_list_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, avg_episode_reward_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list_finetune, avg_episode_reward_std_list_finetune)],
                [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list_finetune, avg_episode_reward_std_list_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Average Episode Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot average trajectory length
    plt.subplot(2, 3, 2)
    plot_func(num_denoising_steps_list_pretrain, avg_traj_length_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain)],
                [avg_best + std for avg_best, std in zip(avg_traj_length_list_pretrain, avg_traj_length_list_std_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, avg_traj_length_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(avg_traj_length_list_finetune, avg_traj_length_list_std_finetune)],
                [avg_best + std for avg_best, std in zip(avg_traj_length_list_finetune, avg_traj_length_list_std_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Average Trajectory Length')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Trajectory Length')
    
    plt.grid(True)
    plt.legend()
    
    # Plot average best reward with shading
    plt.subplot(2, 3, 4)
    plot_func(num_denoising_steps_list_pretrain, avg_best_reward_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(avg_best_reward_list_pretrain, avg_best_reward_std_list_pretrain)],
                [avg_best + std for avg_best, std in zip(avg_best_reward_list_pretrain, avg_best_reward_std_list_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, avg_best_reward_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(avg_best_reward_list_finetune, avg_best_reward_std_list_finetune)],
                [avg_best + std for avg_best, std in zip(avg_best_reward_list_finetune, avg_best_reward_std_list_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Average Best Reward')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Average Best Reward')
    plt.grid(True)
    plt.legend()
    
    # Plot success rate
    plt.subplot(2, 3, 5)
    plot_func(num_denoising_steps_list_pretrain, success_rate_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [avg_best - std for avg_best, std in zip(success_rate_list_pretrain, success_rate_list_std_pretrain)],
                [avg_best + std for avg_best, std in zip(success_rate_list_pretrain, success_rate_list_std_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, success_rate_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [avg_best - std for avg_best, std in zip(success_rate_list_finetune, success_rate_list_std_finetune)],
                [avg_best + std for avg_best, std in zip(success_rate_list_finetune, success_rate_list_std_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Success Rate')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()

    # Plot inference frequency
    plt.subplot(2, 3, 3)
    plot_func(num_denoising_steps_list_pretrain, avg_single_step_freq_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [freq - std for freq, std in zip(avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain)],
                [freq + std for freq, std in zip(avg_single_step_freq_list_pretrain, avg_single_step_freq_std_list_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, avg_single_step_freq_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [freq - std for freq, std in zip(avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune)],
                [freq + std for freq, std in zip(avg_single_step_freq_list_finetune, avg_single_step_freq_std_list_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Inference Frequency')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference Frequency')
    plt.grid(True)
    plt.legend()
    
    # Plot inference frequency
    plt.subplot(2, 3, 6)
    plot_func(num_denoising_steps_list_pretrain, avg_single_step_duration_list_pretrain, marker='o', label=label_1, color=color_1)
    plt.fill_between(num_denoising_steps_list_pretrain,
                [duration - std for duration, std in zip(avg_single_step_duration_list_pretrain, avg_single_step_duration_std_list_pretrain)],
                [duration + std for duration, std in zip(avg_single_step_duration_list_pretrain, avg_single_step_duration_std_list_pretrain)],
                color=color_1, alpha=0.2)
    plot_func(num_denoising_steps_list_finetune, avg_single_step_duration_list_finetune, marker='o', label=label_2, color=color_2)
    plt.fill_between(num_denoising_steps_list_finetune,
                [duration - std for duration, std in zip(avg_single_step_duration_list_finetune, avg_single_step_duration_std_list_finetune)],
                [duration + std for duration, std in zip(avg_single_step_duration_list_finetune, avg_single_step_duration_std_list_finetune)],
                color=color_2, alpha=0.2)
    if add_denoise_step_line: plt.axvline(x=inference_steps,  color=color_2, linestyle='--', label=f'Step={inference_steps}')
    plt.title('Inference duration')
    plt.xlabel('Number of Denoising Steps')
    plt.ylabel('Inference duration')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(f"{model_name}, {env_name} \n steps = {', '.join(map(str, num_denoising_steps_list_pretrain))}", fontsize=25)
    plt.tight_layout()
    fig_path = os.path.join(log_dir, SAVE_FILE_NAME)
    plt.savefig(fig_path)
    print(f"figure saved to {fig_path}")
    plt.close()
    

if __name__ == '__main__':
    MODEL_NAME='DiffusionModel' #'Compare' #'DiffusionModel' #'ReFlow'
    FINETUNE_STEP=20 #4
    LABEL1='Pretrained' #'ReinFlow(learnable)'  #'Pretrained'
    COLOR1='black' #'red'
    LABEL2='Finetuned' #'DPPO'  #'Finetuned'
    COLOR2='red' #'black'
    ADD_DENOISE_STEP_LINE = True #False  # 
    SAVE_FILE_NAME='compare.png'
    PLOT_SCALE='standard'
    HOPPER=[
        #     'visualize/flow/25-01-30-21-27-04/eval_statistics.npz',
                # 'visualize/0_Models/ReFlow/hopper-medium-v2/25-02-22-20-55-20_uniform_16timedim_pretrain/eval_statistics.npz',
                'visualize/0_Models/DiffusionModel/hopper-medium-v2/25-02-07-15-15-46_pretrained/eval_statistics.npz',
                #     'visualize/flow/25-01-30-21-27-22/eval_statistics.npz',
                # 'visualize/0_Models/ReFlow/hopper-medium-v2/25-02-22-20-53-02_uniform_16timedim_RLFT_at_step=4/eval_statistics.npz',
                # 'visualize/0_Models/ReFlow/hopper-medium-v2/25-02-24-10-42-59_uniform_16timedim_RLFT_at_step4_learnable/eval_statistics.npz',
                'visualize/0_Models/DiffusionModel/hopper-medium-v2/25-02-22-21-51-29_RLfinetuned/eval_statistics.npz',
                'hopper-v2'
            ]   #hopper-medium-v2
    
    HALFCHEETAH=[
            'visualize/flow/25-02-02-00-24-33/eval_statistics.npz',
                 'visualize/flow/25-02-02-00-26-14/eval_statistics.npz',
                 'halfcheetah-v2'
                 ] # halfcheetah-medium-v2
    
    WALKER=[
            'visualize/flow/25-02-02-00-23-14/eval_statistics.npz',
            'visualize/flow/25-02-02-00-23-10/eval_statistics.npz',
            'walker2d-v2'
            ] # walker2d-medium-v2
    
    ANT = [
            'visualize/flow/25-02-02-00-32-29/eval_statistics.npz',
           'visualize/flow/25-02-02-00-34-27/eval_statistics.npz',
           'ant-v0'
           ] #ant-medium-expert-v0
    
    pretrain_eval_path, finetune_eval_path, ENV_NAME= HOPPER #ANT
    
    pretrain_file = read_eval_statistics(pretrain_eval_path)
    
    finetune_file = read_eval_statistics(finetune_eval_path)
    
    logdir = os.path.join('agent','eval','visualize', MODEL_NAME, 'pretrain_finetune_compare', ENV_NAME, f'{current_time()}')
    os.makedirs(logdir, exist_ok=True)
    
    plot_eval_statistics(pretrain_file, finetune_file, 
                         FINETUNE_STEP, 
                         MODEL_NAME, 
                         ENV_NAME, 
                         LABEL1, COLOR1,
                         LABEL2, COLOR2, 
                         ADD_DENOISE_STEP_LINE, 
                         logdir,
                         SAVE_FILE_NAME,
                         PLOT_SCALE)