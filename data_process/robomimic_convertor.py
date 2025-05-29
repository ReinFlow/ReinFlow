


"""
This file converts the official robomimic dataset to our dataset format.
"""





import h5py
import numpy as np
import os

# Define paths (adjust as needed if downloading from a URL is required)
hdf5_path = "data/robomimic/transport/state/low_dim_v15.hdf5?download=true"  # Assuming local file; modify if URL
output_dir = "data/robomimic/transport-state-PH"  # Output directory for NPZ files

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define path for summary log
log_path = os.path.join(output_dir, 'summary.log')

# Function to log messages to both console and file
def log_message(message):
    print(message)
    with open(log_path, 'a') as log_file:
        log_file.write(message + '\n')

# Open the HDF5 file and process the data
with h5py.File(hdf5_path, 'r') as f:
    data_group = f['data']
    
    # Get all trajectory keys (e.g., 'demo_0', 'demo_1', etc.)
    traj_keys = [key for key in data_group.keys() if key.startswith('demo_')]
    
    # Initialize lists to collect data from all trajectories
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    traj_lengths = []
    
    # Iterate through each trajectory
    for traj_key in traj_keys:
        traj_group = data_group[traj_key]
        
        # Extract number of samples in the trajectory
        num_samples = traj_group.attrs['num_samples']
        traj_lengths.append(num_samples)
        
        # Extract datasets (using 'states' for state-based inputs as per task)
        states = traj_group['states'][:]       # Shape: (N, D)
        actions = traj_group['actions'][:]     # Shape: (N, A)
        rewards = traj_group['rewards'][:]     # Shape: (N,)
        dones = traj_group['dones'][:]         # Shape: (N,)
        
        # Append to lists
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_dones.append(dones)
    
    # Concatenate all trajectories into single arrays
    states = np.concatenate(all_states, axis=0)      # Shape: (total_steps, state_dim)
    actions = np.concatenate(all_actions, axis=0)    # Shape: (total_steps, action_dim)
    rewards = np.concatenate(all_rewards, axis=0)    # Shape: (total_steps,)
    dones = np.concatenate(all_dones, axis=0)        # Shape: (total_steps,)
    traj_lengths = np.array(traj_lengths)            # Shape: (num_trajectories,)
    
    # Verify data consistency
    total_steps = states.shape[0]
    assert total_steps == np.sum(traj_lengths), "Total steps do not match sum of trajectory lengths"
    assert actions.shape[0] == total_steps, "Actions length mismatch"
    assert rewards.shape[0] == total_steps, "Rewards length mismatch"
    assert dones.shape[0] == total_steps, "Dones length mismatch"
    
    # Compute normalization statistics for states and actions
    obs_min = states.min(axis=0)         # Min per state dimension
    obs_max = states.max(axis=0)         # Max per state dimension
    action_min = actions.min(axis=0)     # Min per action dimension
    action_max = actions.max(axis=0)     # Max per action dimension
    
    # Normalize states and actions to [-1, 1] range
    states_normalized = 2 * (states - obs_min) / (obs_max - obs_min + 1e-6) - 1
    actions_normalized = 2 * (actions - action_min) / (action_max - action_min + 1e-6) - 1
    
    # Define output file paths
    normalization_path = os.path.join(output_dir, 'normalization.npz')
    train_path = os.path.join(output_dir, 'train.npz')
    
    # Save normalization statistics
    np.savez(
        normalization_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max
    )
    log_message(f"Normalization file saved to {normalization_path}")
    
    # Save training data (normalized states and actions, plus raw rewards, terminals, and traj_lengths)
    np.savez(
        train_path,
        states=states_normalized,
        actions=actions_normalized,
        rewards=rewards,
        terminals=dones,  # 'terminals' key matches StitchedSequenceQLearningDataset expectation
        traj_lengths=traj_lengths
    )
    log_message(f"Train file saved to {train_path}")
    
    # Log shapes and other information
    log_message(f"total_steps={total_steps}")
    log_message(f"States shape: {states.shape}")
    log_message(f"Actions shape: {actions.shape}")
    log_message(f"Rewards shape: {rewards.shape}")
    log_message(f"Terminals shape: {dones.shape}")
    log_message(f"Traj_lengths shape: {traj_lengths.shape}")
    log_message(f"Number of trajectories: {traj_lengths.shape[0]}")
    log_message(f"Processing summary saved to: {log_path}")