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


# Humanoid expert demonstration collector (SAC agent) 
# Revised from https://github.com/cubrink/mujoco-2.1-rl-project
# put this file to the experiments/sac-humanoid-v3-exp0 in the repository  https://github.com/cubrink/mujoco-2.1-rl-project.git and run the script, 
# you can collect an expert dataset of a Humanoid-v3 agent trained by SAC. 

import os
import sys
from pathlib import Path
# Path hack as we don't need this to be an installable package
root_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # ant-v3-exp1  # experiments
)  # mujoco-2.1-rl-project
sys.path.insert(0, root_path)

import gym
import torch
from rl.algorithms.sac import SAC
import numpy as np
from tqdm import tqdm as tqdm 
def sample_dataset(sac, env, num_trajectories=600, max_steps=500, file_path='humanoid_expert_dataset.npz'):
    """
    Collect an expert dataset from an SAC agent in the Humanoid-v3 environment and save it as an .npz file.
    
    Parameters:
    - sac: The trained SAC agent.
    - env: The Humanoid-v3 environment.
    - num_trajectories: Number of trajectories to collect (default: 600).
    - max_steps: Maximum steps per trajectory (default: 500, matching env._max_episode_steps).
    - file_path: Path to save the .npz file (default: 'humanoid_expert_dataset.npz').
    
    Returns:
    - None (saves dataset to file).
    """
    traj_states_list = []
    traj_actions_list = []
    traj_lengths = []
    
    for _ in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        state_curr = env.reset()
        traj_states = []
        traj_actions = []
        stationary_count = 0
        
        for _ in tqdm(range(max_steps)):
            action = sac.get_action(state_curr, sample_from_dist=False)
            state_next, _, done, _ = env.step(action)
            traj_states.append(state_curr)
            traj_actions.append(action)
            
            # Stationary check as in the test method
            mse = np.sum((state_curr - state_next) ** 2)
            if mse < sac.mse_threshold:
                stationary_count += 1
            else:
                stationary_count = 0
            
            if stationary_count >= 50:  # Hardcoded threshold from test method
                done = True
            
            state_curr = state_next
            if done:
                break
        
        # Convert trajectory data to numpy arrays with correct dtype
        traj_states = np.array(traj_states, dtype=np.float32)  # Shape: (traj_len, 376)
        traj_actions = np.array(traj_actions, dtype=np.float32)  # Shape: (traj_len, 17)
        traj_lengths.append(len(traj_states))
        traj_states_list.append(traj_states)
        traj_actions_list.append(traj_actions)
    
    # Concatenate all trajectories
    all_states = np.concatenate(traj_states_list, axis=0)  # Shape: (total_time_steps, 376)
    all_actions = np.concatenate(traj_actions_list, axis=0)  # Shape: (total_time_steps, 17)
    traj_lengths = np.array(traj_lengths, dtype=np.int64)  # Shape: (num_trajectories,)
    
    # Save to .npz file
    np.savez(file_path, states=all_states, actions=all_actions, traj_lengths=traj_lengths)
    print(f"Dataset saved to {file_path} with {len(traj_lengths)} trajectories and {all_states.shape[0]} time steps.")

if __name__ == "__main__":
    model_dir = Path(__file__).parent / "models"
    env = gym.make("Humanoid-v3",)
    env._max_episode_steps = 500
    sac = SAC(
        env,
        hidden_sizes=(256, 256),
        update_freq=64,
        num_update=64,
        update_threshold=4096,
        batch_size=128,
        model_dir=model_dir,
        alpha=0.5,
        device="cuda:0",
    )
    models_to_test = ["sac-15278-370000.pt"]                            #"sac-5396-260000.pt"] #"sac-2514-240000.pt",  "sac-15278-370000"
    data=torch.load(model_dir / models_to_test[0], weights_only=True)
    print(f"loaded successfully data")
    sac.load_state_dict(data)
    print(f"loaded successfully agent")
    
    sample_dataset(
        sac=sac, 
        env=env, 
        num_trajectories=512,
        max_steps=500,
        file_path= 'experiments/sac-humanoid-v3-exp0/data/humanoid_expert_dataset.npz'
    )
