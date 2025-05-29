
## Recreating All the Plots in Our Paper

- **Preparation:** Grab all files from the `visualize` branch in the ReinFlow repo and unzip to `<REINFLOW_DIR>/visualize/`.

### Figure 1: Wall Time Efficiency in OpenAI Gym
```bash
# gym-hopper
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=AverageEpisodeReward environment_name=gym-state task_name=hopper-d4rl
# gym-walker2d
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=AverageEpisodeReward environment_name=gym-state task_name=walker-d4rl
# gym-ant-v2
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=AverageEpisodeReward environment_name=gym-state task_name=ant-d4rl
# gym-Humanoid-v3
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=AverageEpisodeReward environment_name=gym-state task_name=humanoid-d4rl
# the legend
python agent/eval/visualize/crop_pdfs.py --input_pdf=/visualize/Final_experiments/outs/gym-state_hopper-d4rl_AverageEpisodeReward_legend.pdf --output_pdf=/visualize/Final_experiments/outs/gym-state_hopper-d4rl_AverageEpisodeReward_legend_crop.pdf --left_percent=20 --right_percent=20
```

### Figure 2: Task Completion Rates in Franka Kitchen
```bash
# kitchen-complete-v0
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=TaskCompletionRate environment_name=kitchen task_name=kitchen-complete-v0
# kitchen-mixed-v0
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=TaskCompletionRate environment_name=kitchen task_name=kitchen-mixed-v0
# kitchen-partial-v0
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=TaskCompletionRate environment_name=kitchen task_name=kitchen-partial-v0
# the legend
python agent/eval/visualize/crop_pdfs.py --input_pdf=visualize/Final_experiments/outs/kitchen_kitchen-complete-v0_TaskCompletionRate_legend.pdf --output_pdf=visualize/Final_experiments/outs/kitchen_kitchen-complete-v0_TaskCompletionRate_legend_crop.pdf --left_percent=30 --right_percent=30 --top_percent=10 --bottom_percent=10
```

### Figure 3: Success Rates in Robomimic Visual Manipulation Tasks
```bash
# robomimic-can
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=SuccessRate environment_name=robomimic-img task_name=can-img
# robomimic-square
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=SuccessRate environment_name=robomimic-img task_name=square-img
# robomimic-transport
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=SuccessRate environment_name=robomimic-img task_name=transport-img
# for the legend
python agent/eval/visualize/crop_pdfs.py --input_pdf=visualize/Final_experiments/outs/robomimic-img_can-img_SuccessRate_legend.pdf --output_pdf=visualize/Final_experiments/outs/robomimic-img_can-img_SuccessRate_legend_crop.pdf --left_percent=20 --right_percent=20 --top_percent=10 --bottom_percent=10
```

### Figure 4 (a): ReFlow Policy in Hopper-v2
```bash
cd ~/ReinFlow
python agent/eval/visualize/compare_pretrain_finetune_data_denoise_step.py
python agent/eval/visualize/crop_pdfs.py input_pdf=<path_to_your_generated_pdf> output_pdf=<path_to_your_output_pdf>
```

### Figure 4(b): Shortcut Policy in Square
```bash
cd ~/ReinFlow
python agent/eval/visualize/compare_pretrain_finetune_data.py
```

### Figure 4(c): ReFlow Policy in Square
```bash
cd ~/ReinFlow
python agent/eval/visualize/success_rate_episode_reward.py environment_name=robomimic-img task_name=square-img-logitbeta evaluation_name=SuccessRate  
```

### Figure 5(a): Noise Input’s Effect in Ant
```bash
python agent/eval/visualize/sensitivity_sigma_entropy.py
```

### Figure 5(b): Noise Condition’s Effect in Kitchen-partial
```bash
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=TaskCompletionRate environment_name=kitchen task_name=kitchen-partial-v0-sigma_s_t 
```

### Figure 6(a): Noise Level Affects Exploration in "Ant"
```bash
python agent/eval/visualize/sensitivity_sigma_entropy.py
```

### Figure 6(b): Regularization Affects ReinFlow in Humanoid-v3
```bash
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=AverageEpisodeReward environment_name=gym-state task_name=humanoid-regularize-compare
```

### Figure 7: Fine-tuning a Flow Matching Policy with Online RL Algorithm ReinFlow
This is a schematic plotted by software.

### Figure 8: Four OpenAI Gym Locomotion Tasks: Hopper, Walker2d, Ant, and Humanoid
These figures are screenshots from the MuJoCo simulator.

### Figure 9: Four Manipulation Tasks in State-Input Franka Kitchen and Pixel-Input Robomimic Environments
These figures are screenshots from the MuJoCo simulator.

### Figure 10: Sample Efficiency Results of State-Based Locomotion Tasks in OpenAI Gym
```bash
# these figures will be generated automatically when you plot Figure 1. 
```

### Figure 11: Fine-tuning Locomotion Task Ant with Diffusion RL Baselines and ReinFlow
```bash
# hopper
TASK_NAME=hopper
python agent/eval/visualize/average_episode_reward_all_baselines.py --config-dir=agent/eval/visualize/visualize_cfgs/ --config-name=final_experiments environment_name=gym-state task_name=${TASK_NAME} env.gym-state.${TASK_NAME}.csv_filename=${TASK_NAME}_reward_shortcut_correct.csv evaluation_name=AverageEpisodeReward output_filename=all_diffusion_baselines_${TASK_NAME}
```

### Figure 12: Fine-tuning Locomotion Task Hopper-v2 and Walker2d-v2 with Diffusion RL Baselines and ReinFlow
```bash
# walker2d
TASK_NAME=walker
python agent/eval/visualize/average_episode_reward_all_baselines.py --config-dir=agent/eval/visualize/visualize_cfgs/ --config-name=final_experiments environment_name=gym-state task_name=${TASK_NAME} env.gym-state.${TASK_NAME}.csv_filename=${TASK_NAME}_reward_shortcut_correct.csv evaluation_name=AverageEpisodeReward output_filename=all_diffusion_baselines_${TASK_NAME}
# ant
TASK_NAME=ant
python agent/eval/visualize/average_episode_reward_all_baselines.py --config-dir=agent/eval/visualize/visualize_cfgs/ --config-name=final_experiments environment_name=gym-state task_name=${TASK_NAME} env.gym-state.${TASK_NAME}.csv_filename=${TASK_NAME}_reward_shortcut_correct.csv evaluation_name=AverageEpisodeReward output_filename=all_diffusion_baselines_${TASK_NAME}
```

### Figure 13: Fine-tuning Shortcut Policy in Kitchen-complete-v0 at Different Denoising Steps
```bash
python agent/eval/visualize/success_rate_episode_reward.py evaluation_name=TaskCompletionRate task_name=kitchen-complete-v0-denoise_step
```
