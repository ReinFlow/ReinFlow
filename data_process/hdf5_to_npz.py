"""
This file converts hdf5 to npz files.
"""

import h5py
import numpy as np
import os
from tqdm import tqdm as tqdm
import random
from copy import deepcopy
import argparse
import sys

# Custom print function to log to both console and file
class Logger:
    def __init__(self, log_file):
        self.console = sys.stdout
        self.log_file = open(log_file, 'w')

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def calculate_average_episode_reward(rewards, done_indices):
    """Calculate the average episode reward from the rewards array and done indices."""
    if rewards is None or len(rewards) == 0:
        return None
    
    episode_rewards = []
    prev_index = 0
    for cur_index in done_indices:
        # Sum rewards for the current episode
        episode_reward = np.sum(rewards[prev_index:cur_index + 1])
        episode_rewards.append(episode_reward)
        prev_index = cur_index + 1
    
    # Handle the last episode if it exists
    if prev_index < len(rewards):
        episode_reward = np.sum(rewards[prev_index:])
        episode_rewards.append(episode_reward)
    
    if not episode_rewards:
        return None
    
    return np.mean(episode_rewards)

def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 dataset to NPZ and normalize")
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the input HDF5 dataset file')
    parser.add_argument('--max_episodes', type=int, default=1500,
                        help='Maximum number of episodes to process (0 for all)')
    parser.add_argument('--val_split', type=float, default=0.0,
                        help='Fraction of data to use for validation (0.0 to 1.0)')
    args = parser.parse_args()

    file_path = args.data_path
    save_dir = os.path.dirname(file_path)
    max_episodes = args.max_episodes
    val_split = args.val_split

    # Set up logging to description.log in save_dir
    log_file_path = os.path.join(save_dir, "description.log")
    logger = Logger(log_file_path)
    sys.stdout = logger

    try:
        with h5py.File(file_path, 'r') as f:
            # Extract the states and actions
            states = f['observations'][:]
            actions = f['actions'][:]
            rewards = f['rewards'][:] if 'rewards' in f else None
            # Extract the terminal and timeout conditions
            terminals = f['terminals'][:]
            timeouts = f['timeouts'][:]

            done = np.logical_or(terminals, timeouts)

            # Find the indices where the 'done' array changes from 0 to 1
            episode_start_indices = np.where(np.diff(done.astype(int)) == 1)[0] + 1

            # Calculate the episode lengths
            first_episode_length = [episode_start_indices[0]]
            second_to_last_episode_length = episode_start_indices[1:] - episode_start_indices[:-1]
            traj_lengths = np.concatenate((first_episode_length, second_to_last_episode_length))

            # Print the shapes of the arrays
            print(f"Raw data:")
            print("States shape:", states.shape)
            print("Actions shape:", actions.shape)
            print("Done shape:", done.shape)
            print("traj_lengths shape:", traj_lengths.shape)
            idx = 0
            print(f"idx={idx}")
            print(f"terminals[]={terminals[idx]}")
            print(f"timeouts[]={timeouts[idx]}")

            # Calculate and print average episode reward if rewards exist
            if rewards is not None:
                avg_episode_reward = calculate_average_episode_reward(rewards, episode_start_indices)
                if avg_episode_reward is not None:
                    print(f"Average episode reward: {avg_episode_reward}")
                else:
                    print("No episodes found to calculate average episode reward.")
            else:
                print("Rewards attribute not found in dataset.")

            ######################################### clean the data ####################################
            dataset = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'terminals': terminals,
                'timeouts': timeouts
            }
            print("\n========== Basic Info ===========")
            print(f"Keys in the dataset: {dataset.keys()}")
            print(f"State shape: {dataset['states'].shape}")
            print(f"Action shape: {dataset['actions'].shape}")

            # determine trajectories from terminals and timeouts
            terminal_indices = np.argwhere(dataset["terminals"])[:, 0]
            timeout_indices = np.argwhere(dataset["timeouts"])[:, 0]
            done_indices = np.sort(np.concatenate([terminal_indices, timeout_indices]))
            traj_lengths = np.diff(np.concatenate([[0], done_indices + 1]))

            obs_min = np.min(dataset["states"], axis=0)
            obs_max = np.max(dataset["states"], axis=0)
            action_min = np.min(dataset["actions"], axis=0)
            action_max = np.max(dataset["actions"], axis=0)

            print(f"Total transitions: {np.sum(traj_lengths)}")
            print(f"Total trajectories: {len(traj_lengths)}")
            print(
                f"Trajectory length mean/std: {np.mean(traj_lengths)}, {np.std(traj_lengths)}"
            )
            print(
                f"Trajectory length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
            )
            print(f"obs min: {obs_min} \n obs max: {obs_max}")
            print(f"action min: {action_min} \n action max: {action_max}")
            
            # Subsample episodes if needed
            if max_episodes > 0:
                traj_lengths = traj_lengths[: max_episodes]
                done_indices = done_indices[: max_episodes]

            # Split into train and validation sets
            num_traj = len(traj_lengths)
            num_train = int(num_traj * (1 - val_split))
            train_indices = random.sample(range(num_traj), k=num_train)

            # Prepare data containers for train and validation sets
            out_train = {
                "states": [],
                "actions": [],
                "rewards": [],
                "terminals": [],
                "traj_lengths": [],
            }
            out_val = deepcopy(out_train)
            prev_index = 0
            train_episode_reward_all = []
            val_episode_reward_all = []
            for i, cur_index in tqdm(enumerate(done_indices), total=len(done_indices)):
                if i in train_indices:
                    out = out_train
                    episode_reward_all = train_episode_reward_all
                else:
                    out = out_val
                    episode_reward_all = val_episode_reward_all

                # Get the trajectory length and slice
                traj_length = cur_index - prev_index + 1
                trajectory = {
                    key: dataset[key][prev_index : cur_index + 1]
                    for key in ["states", "actions", "rewards", "terminals"]
                }

                # Skip if there is no reward in the episode
                if rewards is not None and np.sum(trajectory["rewards"]) > 0:
                    # Scale observations and actions
                    trajectory["states"] = (
                        2 * (trajectory["states"] - obs_min) / (obs_max - obs_min + 1e-6) - 1
                    )
                    trajectory["actions"] = (
                        2
                        * (trajectory["actions"] - action_min)
                        / (action_max - action_min + 1e-6)
                        - 1
                    )

                    for key in ["states", "actions", "rewards", "terminals"]:
                        out[key].append(trajectory[key])
                    out["traj_lengths"].append(traj_length)
                    episode_reward_all.append(np.sum(trajectory["rewards"]))
                else:
                    print(f"Skipping trajectory {i} due to zero rewards.")

                prev_index = cur_index + 1
            
            # Concatenate trajectories
            for key in ["states", "actions", "rewards", "terminals"]:
                out_train[key] = np.concatenate(out_train[key], axis=0)

                # Only concatenate validation set if it exists
                if val_split > 0:
                    out_val[key] = np.concatenate(out_val[key], axis=0)

            # Save train dataset to npz files
            train_save_path = os.path.join(save_dir, "train.npz")
            np.savez_compressed(
                train_save_path,
                states=np.array(out_train["states"]),
                actions=np.array(out_train["actions"]),
                rewards=np.array(out_train["rewards"]),
                terminals=np.array(out_train["terminals"]),
                traj_lengths=np.array(out_train["traj_lengths"]),
            )
            print(f"train saved to {train_save_path}")

            # Save validation dataset to npz files
            val_save_path = os.path.join(save_dir, "val.npz")
            if val_split > 0.0:
                np.savez_compressed(
                    val_save_path,
                    states=np.array(out_val["states"]),
                    actions=np.array(out_val["actions"]),
                    rewards=np.array(out_val["rewards"]),
                    terminals=np.array(out_val["terminals"]),
                    traj_lengths=np.array(out_val["traj_lengths"]),
                )
            print(f"validation set saved to {val_save_path}")
            
            normalization_save_path = os.path.join(
                save_dir, "normalization.npz"
            )
            np.savez(
                normalization_save_path,
                obs_min=obs_min,
                obs_max=obs_max,
                action_min=action_min,
                action_max=action_max,
            )
            print(f"normalization saved to {normalization_save_path}")
            print(f"dataset statistics saved to {log_file_path}")
            
            # Logging summary statistics
            print("\n========== Final ===========")
            print(
                f"Train - Trajectories: {len(out_train['traj_lengths'])}, Transitions: {np.sum(out_train['traj_lengths'])}"
            )
            print(
                f"Val - Trajectories: {len(out_val['traj_lengths'])}, Transitions: {np.sum(out_val['traj_lengths'])}"
            )
            print(
                f"Train - Mean/Std trajectory length: {np.mean(out_train['traj_lengths'])}, {np.std(out_train['traj_lengths'])}"
            )
            (
                print(
                    f"Val - Mean/Std trajectory length: {np.mean(out_val['traj_lengths'])}, {np.std(out_val['traj_lengths'])}"
                )
                if val_split > 0
                else None
            )
            if avg_episode_reward:
                print(f"avg_episode_reward: {avg_episode_reward}")
            else:
                print(f"This is a pure behavior cloning dataset, no 'rewards' attribute found. ")
    finally:
        # Ensure the log file is properly closed
        sys.stdout = logger.console
        logger.close()

if __name__ == "__main__":
    main()