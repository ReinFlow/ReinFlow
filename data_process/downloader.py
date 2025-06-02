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
import argparse
import gym
import numpy as np
import collections
import pickle
from tqdm import tqdm
import d4rl

def download_d4rl_mujoco_data(env_name, dataset_type):
	datasets = []

	data_dir = 'data/'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	assert env_name in ['walker2d', 'halfcheetah', 'hopper']
	assert dataset_type in ['medium', 'medium-expert', 'medium-replay', 'random',
                         'expert']

	name = f'{env_name}-{dataset_type}-v2'
	pkl_file_path = os.path.join(data_dir, name)

	print("processing: ", name)

	env = gym.make(name)
	dataset = env.get_dataset()

	N = dataset['rewards'].shape[0]
	data_ = collections.defaultdict(list)

	use_timeouts = False
	if 'timeouts' in dataset:
		use_timeouts = True

	episode_step = 0
	paths = []
	for i in range(N):
		done_bool = bool(dataset['terminals'][i])
		if use_timeouts:
			final_timestep = dataset['timeouts'][i]
		else:
			final_timestep = (episode_step == 1000-1)
		for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
			data_[k].append(dataset[k][i])
		if done_bool or final_timestep:
			episode_step = 0
			episode_data = {}
			for k in data_:
				episode_data[k] = np.array(data_[k])
			paths.append(episode_data)
			data_ = collections.defaultdict(list)
		episode_step += 1

	returns = np.array([np.sum(p['rewards']) for p in paths])
	num_samples = np.sum([p['rewards'].shape[0] for p in paths])
	print(f'Number of samples collected: {num_samples}')
	print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

	np.save(f'{pkl_file_path}.npy', paths)
	with open(f'{pkl_file_path}.pkl', 'wb') as f:
		pickle.dump(paths, f)
    


if __name__=="__main__":
    download_d4rl_mujoco_data(env_name='hopper', dataset_type='medium')