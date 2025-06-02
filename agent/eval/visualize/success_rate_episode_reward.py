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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns 
import re
import hydra
from omegaconf import OmegaConf
from util.dirs import REINFLOW_DIR
from agent.eval.visualize.constants import method_name_dict, max_n_step_dict, time_step_ratios


PPO_EVAL_FREQ=10
FQL_EVAL_FREQ=5000
FQL_LOG_FREQ=200



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
    if 'kitchen' in task_name:
        evaluation_name='TaskCompletionRate'
        data.loc[:, data.columns != 'Step'] = data.loc[:, data.columns != 'Step'] * 0.25

    if task_name in max_n_step_dict.keys():
        max_n_steps=max_n_step_dict[task_name]
    else:
        raise NotImplementedError

    method_config=method_name_dict[task_name]

    # Calculate x-axis based on plot_x_axis
    if plot_x_axis == 'step':
        x_axis = data['Step'].values
        x_label = 'Steps'
    elif plot_x_axis == 'sample':
        raw_steps = data['Step'].values
        x_axis = raw_steps * n_parallel_envs * n_rollout_steps * n_act_steps
        x_label = 'Samples'
    else:
        raise ValueError(f"Unsupported plot_x_axis: {plot_x_axis}. Must be 'step' or 'sample'.")
    
    # Extract methods and seeds from column names
    method_seed_map = {}
    unmatched_columns = []
    
    for col in data.columns:
        if ('success rate' in col and ('__MIN' not in col) and ('__MAX' not in col)) or \
            ('episode reward' in col and ('__MIN' not in col) and ('__MAX' not in col)):
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
    if unmatched_columns !=[]:
        print(f"unmatched_columns={unmatched_columns}")
    # Define method names and colors

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
        suc_rates = []
        for seed, col in seed_cols:
            rate=data[col].dropna().values # [0:n_evals]
            if 'FQL' ==method:
                print(f"seed={seed}, rate={len(rate)}, {rate}")
            suc_rates.append(rate)
            
        if 'FQL' ==method:
            print(f"**rate={len(rate)}, {rate}")
        min_len=min(len(rate) for rate in suc_rates)
        if 'FQL' ==method:
            print(f"**min_len={min_len}")
        truncated_suc_rates = [rate[0:min_len] for rate in suc_rates]
        suc_rates = np.array(truncated_suc_rates)
        if 'FQL' ==method:
            print(f"**suc_rates={suc_rates.shape}")
            # exit()
            pass
        mean_suc_rate = np.nanmean(suc_rates, axis=0)
        std_suc_rate = np.nanstd(suc_rates, axis=0)
        
        method_stats[method] = {
            'mean': mean_suc_rate,
            'std': std_suc_rate,
            'seeds': [seed for seed, _ in seed_cols]
        }    
    
    # Calculate and print average success rate for ReinFlow-xx methods at first and last steps
    for method in updated_method_seed_map:
        if method.startswith('ReinFlow') and method in method_stats:
            mean_suc_rate = method_stats[method]['mean']
            std_suc_rate = method_stats[method]['std']
            if evaluation_name=='SuccessRate' or evaluation_name=='TaskCompletionRate':
                print(f"\n\n{method} - First step {evaluation_name}: {mean_suc_rate[0]*100:.2f}% \u00B1 {std_suc_rate[0]*100:.2f}%\nLast step {evaluation_name}: {mean_suc_rate[-1]*100:.2f}%\u00B1 {std_suc_rate[-1]*100:.2f}%\nDifference: {(mean_suc_rate[-1]-mean_suc_rate[0])*100:.2f}%\nImprovement ratio: {mean_suc_rate[-1]/mean_suc_rate[0]*100:.2f}%\n\n")
            elif evaluation_name=='AverageEpisodeReward':
                print(f"\n\n{method} - First step {evaluation_name}: {mean_suc_rate[0]:.2f}\u00B1 {std_suc_rate[0]:.2f}\nLast step {evaluation_name}: {mean_suc_rate[-1]:.2f}\u00B1 {std_suc_rate[-1]:.2f}\nDifference: {(mean_suc_rate[-1]-mean_suc_rate[0]):.2f}\nImprovement ratio: {mean_suc_rate[-1]/mean_suc_rate[0]*100:.2f}%\n\n")
            else:
                raise ValueError(f"evaluation_name={evaluation_name} not in ['SuccessRate', 'AverageEpisodeReward']")
    # Set plot style
    sns.set_theme(style='whitegrid')
    figsize=(10, 6)
    if environment_name=='gym-state' and task_name in ['hopper-d4rl', 'walker-d4rl', 'ant-d4rl', 'humanoid-d4rl']:
        figsize=(38,6)
    fig, ax = plt.subplots(figsize=figsize)
    
    color_map = {m['display_name']: m['color'] for m in method_config}
    
    handles, labels = [], []
    ours_methods = [m for m in updated_method_seed_map if '(ours)' in m]
    dppo_method = 'DPPO'
    fql_method = 'FQL'
    
    for method in ours_methods:
        if method in method_stats:
            stats = method_stats[method]
            mean = stats['mean']
            std = stats['std']
            color = color_map[method]
            # Add rim line (thicker, black)
            ax.plot(x_axis[:len(mean)], mean, linewidth=5, color=color)  # Rim line
            line, = ax.plot(x_axis[:len(mean)], mean, label=method, linewidth=3, color=color)
            line.set_linewidth(4)  # Thicken legend line
            ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.1, color=color)
            handles.append(line)
            labels.append(method)
    
    if dppo_method in method_stats:
        stats = method_stats[dppo_method]
        mean = stats['mean']
        std = stats['std']
        color = color_map[dppo_method]
        linewidth = 2
        alpha = 0.6
        # Add rim line (thicker, black)
        ax.plot(x_axis[:len(mean)], mean, linewidth=4, color=color, alpha=alpha)  # Rim line
        line, = ax.plot(x_axis[:len(mean)], mean, label=dppo_method, linewidth=linewidth, color=color, alpha=alpha)
        line.set_linewidth(4)  # Thicken legend line
        ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.12, color=color)
        handles.append(line)
        labels.append(dppo_method)
    
    for method in updated_method_seed_map:
        if method not in ours_methods and method != dppo_method and method != fql_method:
            stats = method_stats[method]
            mean = stats['mean']
            std = stats['std']
            color = color_map[method]
            linewidth = 2
            alpha = 0.6 if method in ['Gaussian', 'DIPO', 'IDQL', 'DQL', 'DRWR', 'DAWR', 'QSM', 'DPPO'] else 1.0
            # Add rim line (thicker, black)
            ax.plot(x_axis[:len(mean)], mean, linewidth=4, color=color, alpha=alpha)  # Rim line
            line, = ax.plot(x_axis[:len(mean)], mean, label=method, linewidth=linewidth, color=color, alpha=alpha)
            line.set_linewidth(4)  # Thicken legend line
            ax.fill_between(x_axis[:len(mean)], mean - std, mean + std, alpha=0.12, color=color)
            handles.append(line)
            labels.append(method)
    
    if fql_method in method_stats:
        stats = method_stats[fql_method]
        mean = stats['mean']
        std = stats['std']
        color = color_map[fql_method]
        linewidth = 2
        alpha = 0.6
        print(f"**raw_steps={raw_steps}, len(raw_steps)={len(raw_steps)}, len(mean)={len(mean)}")
        fql_steps=raw_steps[-len(mean):]
        fql_sample_axis=fql_steps*n_act_steps
        print(f"**fql_steps={fql_steps}, n_act_steps={n_act_steps}, fql_sample_axis={fql_sample_axis}")
        fql_x_axis=fql_sample_axis
        
        # fql_x_axis, mean, std=downsample_and_max_filter(fql_x_axis, mean, std, 10, 5)
        
        if environment_name=='gym-state' and task_name in ['hopper-d4rl', 'walker-d4rl', 'ant-d4rl', 'humanoid-d4rl']: #
            if task_name=='humanoid-d4rl':
                down_sample_rate=2
            else:
                down_sample_rate=5
                
            fql_x_axis=fql_x_axis[::down_sample_rate]
            mean=mean[::down_sample_rate]
            std=std[::down_sample_rate]
        line, = ax.plot(fql_x_axis, mean, label=fql_method, linewidth=linewidth, color=color, alpha=alpha)
        ax.fill_between(fql_x_axis, mean - std, mean + std, alpha=0.6, color=color)
        line.set_linewidth(4)  # Thicken legend line
        handles.append(line)
        labels.append(fql_method)
    
    if task_name =='kitchen-complete-v0':
        # ax.set_xlim(0,max(fql_x_axis))
        pass
    if task_name =='kitchen-partial-v0':
        pass
        ax.set_xlim(0,4.2e6)
    if task_name =='kitchen-mixed-v0':
        pass
        ax.set_xlim(0,2.5e6)
    if task_name =='hopper':
        # ax.set_xlim(0,1e7)
        pass
    # if task_name=='kitchen-partial-v0-sigma_s_t' or task_name=='humanoid-regularize-compare':
    if task_name in ['hopper', 'walker', 'ant', 'humanoid', 'kitchen-partial-v0-sigma_s_t', 'humanoid-regularize-compare', '']:
        bc_value=[]
        for method in updated_method_seed_map:
            bc_value.append(method_stats[method]['mean'][0])
        bc_value=np.array(bc_value).mean()
        bc_line= plt.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=4)
        handles.append(bc_line)
        labels.append('BC')
    
    evaluation_name_spaced=''.join(' ' + c if c.isupper() else c for c in evaluation_name).strip()
    if environment_name=='gym-state' and task_name in ['hopper-d4rl', 'walker-d4rl', 'ant-d4rl', 'humanoid-d4rl']:
        fontsize=42
        # Set y-axis ticks to scientific notation
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # Enlarge the scientific notation exponent to fontsize 22
        ax.yaxis.get_offset_text().set_fontsize(40)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel('Episode Reward', fontsize=fontsize)
    else:
        fontsize=24
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(evaluation_name_spaced, fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # sort handles and labels in a fixed order.
    if 'kitchen' in task_name: # =='kitchen-complete-v0':
        if task_name=='kitchen-complete-v0':
            print(f"Original handles={handles}, labels={labels}")
            # Explicitly sort handles and labels for desired legend order (bottom to top: FQL, DPPO, ReinFlow-S)
            desired_order = ['ReinFlow-S (ours)', 'DPPO','FQL']
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            print(f"Reordered labels={sorted_labels}")
            # ax.legend(sorted_handles, sorted_labels, fontsize=21)
            # ax.set_xlim(0,710000)
        elif task_name=='kitchen-partial-v0-sigma_s_t':
            print(f"Original handles={handles}, labels={labels}")
            # Explicitly sort handles and labels for desired legend order (bottom to top)
            desired_order = [ r'BC',r'ReinFlow-S: $\sigma(s)$', r'ReinFlow-S: $\sigma(s,t)$']
            for label in labels:
                if not label in desired_order:
                    raise ValueError(f"label{label} not in desired_order=[{desired_order}]")
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            print(f"Reordered labels={sorted_labels}")
            # ax.legend(sorted_handles[::-1], sorted_labels[::-1], fontsize=21)
        elif task_name=='kitchen-complete-v0-denoise_step':
            print(f"Original handles={handles}, labels={labels}")
            # Explicitly sort handles and labels for desired legend order (bottom to top)
            desired_order = ['ReinFlow-S (1 step)', 'ReinFlow-S (2 steps)', 'ReinFlow-S (4 steps)']
            for label in labels:
                if not label in desired_order:
                    raise ValueError(f"label{label} not in desired_order=[{desired_order}]")
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            print(f"Reordered labels={sorted_labels}")
            # ax.legend(sorted_handles[::-1], sorted_labels[::-1], fontsize=21)
        else:
            pass
            # ax.legend(handles, labels, fontsize=21) #loc='lower right', bbox_to_anchor=(1.0, 0.2), 
        # ax.set_xlabel('Wall-Clock Time (hours)', fontsize=30)
        # ax.set_ylabel(evaluation_name_spaced, fontsize=27)
    else:
        if task_name=='humanoid-regularize-compare':
            # print(f"Original handles={handles}, labels={labels}")
            # Explicitly sort handles and labels for desired legend order 
            desired_order = [r'$\mathbf{\alpha=0.03}$', r'$\mathbf{\beta=0.01}$', r'$\mathbf{\beta=0.1}$', r'$\mathbf{\beta=1.0}$', r'$\mathbf{BC}$']
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            ax.set_ylim(bottom=800.0)
            # print(f"Reordered labels={sorted_labels}")
            # Set thicker linewidth for legend lines
            for handle in sorted_handles:
                handle.set_linewidth(6.0)  # Increased linewidth for thicker legend lines
            # Place legend on the right, vertically stacked, with short line length
            legend = ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=21, handlelength=1.0)
            # Set legend text to bold
            for text in legend.get_texts():
                text.set_fontweight('bold')
            # Set y-axis ticks to scientific notation
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # Enlarge the scientific notation exponent to fontsize 22
            ax.yaxis.get_offset_text().set_fontsize(22)
        if task_name=='square-img-logitbeta':
            desired_order = ['beta','logitnormal','uniform','BC']
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            print(f"sorted_handles_labels={len(sorted_handles_labels)}")
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            legend = ax.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(1.0, 0.09), fontsize=22) # handlelength=1.0
        elif (environment_name=='robomimic-img' and task_name in ['can-img', 'square-img', 'transport-img']):
            pass
        elif environment_name=='gym-state' and task_name in ['hopper-d4rl', 'walker-d4rl', 'ant-d4rl', 'humanoid-d4rl']:
            pass
        else:
            ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.0, 0.2), fontsize=21)
    ax.xaxis.get_offset_text().set_fontsize(24)  # Enlarge x-axis offset text (e.g., "1e7")
    ax.grid(True)
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
    
    # plot legend.
    # Create a separate figure for the legend, 4 times wider than the original figure
    legend_fig = plt.figure(figsize=(40, 2))  # Width = 10 * 4, height same as original
    legend_ax = legend_fig.add_subplot(111)
    if 'kitchen' in task_name:
        legend = legend_ax.legend(sorted_handles, sorted_labels, fontsize=30, loc='center', ncol=3)
    elif environment_name=='gym-state':
        legend_ax.legend(handles, labels, fontsize=30, loc='center', ncol=4)
    elif environment_name=='robomimic-img':
        desired_order = ['ReinFlow-S (ours)', 'ReinFlow-R (ours)', 'DPPO', 'Gaussian']
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
        print(f"sorted_handles_labels={len(sorted_handles_labels)}")
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        legend_ax.legend(sorted_handles, sorted_labels, fontsize=30, loc='center', ncol=4)
    legend_ax.axis('off')  # Hide axes for the legend figure
    # Save the legend as a separate PDF
    legend_pdf_path = os.path.join(output_dir, f"{output_filename}_legend.pdf")
    legend_fig.savefig(legend_pdf_path, bbox_inches='tight', format='pdf')
    plt.close(legend_fig)  # Close the legend figure to free memory
    # Save the main plot
    # legend_path = os.path.join(output_dir, output_filename)
    # fig.savefig(legend_path, bbox_inches='tight', format='pdf')
    plt.close(fig)  # Close the main figure
    print(f"Legend saved to {legend_pdf_path}")
    
    
    # plot wall-clock-time
    bc_value=[]
    if task_name in time_step_ratios and method in time_step_ratios[task_name]:
        # Create new plot with wall-clock time x-axis, excluding ReinFlow-R
        sns.set_theme(style='whitegrid')
        figsize=(10,6)
        fig, ax = plt.subplots(figsize=figsize)
        handles, labels = [], []
        for method in updated_method_seed_map:
            if method in method_stats and not method.startswith('Gaussian'): # and not method.startswith('ReinFlow-R'):
                stats = method_stats[method]
                mean = stats['mean']
                std = stats['std']
                color = color_map[method]
                if task_name in time_step_ratios and method in time_step_ratios[task_name]:
                    time_ratio = time_step_ratios[task_name][method]
                    # print(f"{method}: mean={mean}, len(mean)={len(mean)}, {min(mean), max(mean)}")
                    data_len=len(mean)
                    if not method=='FQL':
                        eval_interval=PPO_EVAL_FREQ
                        time_per_itr=time_ratio
                    else:
                        eval_interval=FQL_EVAL_FREQ
                        time_per_itr=time_ratio/FQL_LOG_FREQ
                    time_axis = np.arange(data_len) * eval_interval * time_per_itr /3600 # in hours. When we test the time of FQL the interval is measured every 200 iters, so the time elapsed should be devided by 200.                
                # Plot with wall-clock time
                linewidth = 2
                alpha = 0.6 if method in ['FQL', 'Gaussian', 'DIPO', 'IDQL', 'DQL', 'DRWR', 'DAWR', 'QSM', 'DPPO'] else 1.0
                
                if method=='FQL':
                    # down sample FQL
                    print(f"down sample fql for time plot.")
                    time_axis=time_axis[::5]
                    mean=mean[::5]
                    std=std[::5]
                elif method!='Gaussian':
                    print(f"method={method} for BC computing.")
                    bc_value.append(mean[0])
                ax.plot(time_axis, mean, linewidth=4, color=color, alpha=alpha)  # Rim line
                line, = ax.plot(time_axis, mean, label=method, linewidth=linewidth, color=color, alpha=alpha)
                line.set_linewidth(4)  # Thicken legend line
                ax.fill_between(time_axis, mean - std, mean + std, alpha=alpha, color=color)
                handles.append(line)
                labels.append(method)

        #############################################
        if environment_name=='gym-state':
            bc_value=np.array(bc_value).mean()
            bc_line= plt.axhline(y=bc_value, color='black', linestyle='--', label='BC', linewidth=4)
            handles.append(bc_line)
            labels.append('BC')
        #############################################

        ax.set_xlabel('Wall-Clock Time (hours)', fontsize=24)
        ax.set_ylabel(evaluation_name_spaced, fontsize=24)
        ax.tick_params(axis='both', labelsize=24)
        ax.xaxis.get_offset_text().set_fontsize(24)  # Enlarge x-axis offset text
        if 'kitchen' in task_name: #==:
            if task_name=='kitchen-complete-v0':
                print(f"Original handles={handles}, labels={labels}")
                # Explicitly sort handles and labels for desired legend order (bottom to top: FQL, DPPO, ReinFlow-S)
                desired_order = ['ReinFlow-S (ours)', 'DPPO','FQL']
                sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
                sorted_handles, sorted_labels = zip(*sorted_handles_labels)
                print(f"Reordered labels={sorted_labels}")
                # ax.legend(sorted_handles, sorted_labels, fontsize=21)
                # ax.set_xlim(0,1.3)
            else:
                pass
                # ax.legend(handles, labels, fontsize=21) #bbox_to_anchor=(1.0, 0.2),  loc='lower right', 
        else:
            # if task_name =='hopper':
            print(f"Original handles={handles}, labels={labels}")
            # Explicitly sort handles and labels for desired legend order (bottom to top: FQL, DPPO, ReinFlow-S)
            desired_order = ['ReinFlow-S (ours)', 'ReinFlow-R (ours)', 'DPPO','FQL']
            sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)
            print(f"Reordered labels={sorted_labels}")
            if not environment_name=='gym-state':
                ax.legend(sorted_handles, sorted_labels, fontsize=21)
            if environment_name=='gym-state':
                ax.set_xlabel('Wall-Clock Time (hours)', fontsize=30)
                ax.set_ylabel(evaluation_name_spaced, fontsize=27)
                # if 'humanoid' not in task_name:
                #     # ax.legend(sorted_handles, sorted_labels, fontsize=32)
                # else:
                    # ax.legend(sorted_handles, sorted_labels, fontsize=22, loc='upper right')
            # else:
            #     ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.0, 0.2), fontsize=21)
        ax.grid(True)
        # Save wall-clock time plot
        wallclock_output_file_path = os.path.join(fig_dir, f"{output_filename}_wallclock")
        plt.savefig(f'{wallclock_output_file_path}.png', bbox_inches='tight')
        plt.savefig(f'{wallclock_output_file_path}.pdf', bbox_inches='tight')
        print(f"{evaluation_name} wall-clock time comparison saved to {wallclock_output_file_path}.pdf")
        plt.close()

    return updated_method_seed_map

@hydra.main(
    version_base=None,
    config_path=os.path.join(REINFLOW_DIR, "agent/eval/visualize/visualize_cfgs"),
    config_name="final_experiments" # "hopper_datascale_finetune.yaml"
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