

## Adding your own dataset or environment


### Pre-training data

Pre-training script is at [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py). The pre-training dataset [loader](agent/dataset/sequence.py) assumes a npz file containing numpy arrays `states`, `actions`, `images` (if using pixel; img_h = img_w and a multiple of 8) and `traj_lengths`, where `states` and `actions` have the shape of num_total_steps x obs_dim/act_dim, `images` num_total_steps x C (concatenated if multiple images) x H x W, and `traj_lengths` is a 1-D array for indexing across num_total_steps.

<!-- One pre-processing example can be found at [`script/process_robomimic_dataset.py`](script/process_robomimic_dataset.py). -->
<!-- **Note:** The current implementation does not support loading history observations (only using observation at the current timestep). If needed, you can modify [here](agent/dataset/sequence.py#L130-L131). -->

For OpenAI Gym and Franka Kitchen tasks, you can download raw datasets from [D4RL datasets](https://huggingface.co/datasets/imone/D4RL/tree/main), and then run `python data_process/hdf5_to_npz_wrapped.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET>` to convert raw hdf5 to normalized train.npz and normalization.npz files in the same directory. 

To inspect the contents and ranges of the train.npz file, run 
```
python data_process/read_npz.py --data_path=<PATH_TO_YOUR_OFFLINE_RL_DATASET_DIR>/train.npz
```


### Observation history

In our experiments we did not use any observation from previous timesteps (state or pixel), but it is implemented. You can set `cond_steps=<num_state_obs_step>` (and `img_cond_steps=<num_img_obs_step>`, no larger than `cond_steps`) in pre-training, and set the same when fine-tuning the newly pre-trained policy.

### Fine-tuning environment

We follow the Gym format for interacting with the environments. The vectorized environments are initialized at [make_async](env/gym_utils/__init__.py#L10) (called in the parent fine-tuning agent class [here](agent/finetune/train_agent.py#L38-L39)). The current implementation is not the cleanest as we tried to make it compatible with Gym, Robomimic, Furniture-Bench, and D3IL environments, but it should be easy to modify and allow using other environments. We use [multi_step](env/gym_utils/wrapper/multi_step.py) wrapper for history observations and multi-environment-step action execution. We also use environment-specific wrappers such as [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) and [furniture](env/gym_utils/wrapper/furniture.py) for observation/action normalization, etc. You can implement a new environment wrapper if needed.
