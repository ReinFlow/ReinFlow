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


"""
This file normalizes a raw .npz dataset file and outputs a normalization.npz and a normalized training dataset named train.npz
"""




import numpy as np

file_path = None

'''
Arrays in the ant train.npz file:
Array: states
Shape: (2000000, 111)  # 111 = 27 + 84, 84 is the contact force
Data type: float32

Array: actions
Shape: (2000000, 8)
Data type: float32

Array: traj_lengths
Shape: (2409,)
Data type: int64

Arrays in the halfcheetah train.npz file:
['states', 'actions', 'traj_lengths']

Array: states
Shape: (999000, 17)
Data type: float64
Sample data (first 5 rows):

Array: actions
Shape: (999000, 6)
Data type: float64
Sample data (first 5 rows):

Array: traj_lengths
Shape: (999,)
Data type: int64

#######################################################################
Arrays in the /halfcheetah-medium-v2/normalization.npz file:
['obs_min', 'obs_max', 'action_min', 'action_max']

Array: obs_min
Shape: (17,)
Data type: float32
Data content: [ -0.59139955  -3.372963    -0.685813    -0.9440953   -0.65323603
  -1.1924593   -1.3276209   -0.71062547  -2.6428714   -4.630044
  -9.212521   -25.429266   -27.333767   -25.674463   -25.459015
 -28.841232   -26.045986  ]

Array: obs_max
Shape: (17,)
Data type: float32
Data content: [ 0.59435564  3.8933022   1.1060354   0.85662806  0.9202039   0.8702088
  0.9907072   0.6925802   8.634526    3.725341    8.688383   21.009449
 25.731596   20.930292   27.645576   29.017124   25.24359   ]

Array: action_min
Shape: (6,)
Data type: float32
Data content: [-0.9999933 -0.9999983 -0.9999607 -0.9999986 -1.        -0.9999585]

Array: action_max
Shape: (6,)
Data type: float32
Data content: [0.9999996 0.9999481 0.9999882 1.        0.9999933 0.9999995]

'''
# Load the .npz file
with np.load(file_path) as data:
    # Print the keys (names of the arrays stored in the file)
    print("Arrays in the .npz file:")
    print(data.files)

    states = data['states']
    obs_min = states.min(axis=0) 
    obs_max = states.max(axis=0) 
    
    actions = data['actions']
    action_min = actions.min(axis=0) 
    action_max = actions.max(axis=0) 
    
    import os
    normalize_path = os.path.join(os.path.dirname(file_path),'normalization.npz') 
    np.savez(normalize_path, obs_min=obs_min, obs_max=obs_max, action_min=action_min, action_max=action_max)
    print(f"normalization file saved to {normalize_path}\n")
    
    train_path = os.path.join(os.path.dirname(file_path),'train.npz') 
    states_normalize = 2* (states - obs_min ) /( obs_max - obs_min + 1e-6) -1
    actions_normalize = 2* (actions - action_min ) /( action_max - action_min + 1e-6) -1
    np.savez(train_path, states=states_normalize, actions=actions_normalize, traj_lengths=data['traj_lengths'])
    print(f"train file saved to {train_path}\n")
    