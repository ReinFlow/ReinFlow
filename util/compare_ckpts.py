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
Test if two .pt files are the same.
"""


import torch

# Paths to the checkpoint files
pth1 = 'log/gym/pretrain/Humanoid-v3_pre_reflow_mlp_ta4_td20/2025-05-01_18-18-08_42/checkpoint/state_50.pt'
pth2 = 'log/gym/finetune/Humanoid-medium-v3_ppo_reflow_mlp_ta4_td4_tdf4/2025-05-13_00-32-45_seed0/checkpoint/best.pt'
device = 'cuda:4'

# Load the checkpoint dictionaries
data1 = torch.load(pth1, map_location=device, weights_only=True)
data2 = torch.load(pth2, map_location=device, weights_only=True)
print(f"data1['ema'].keys()={data1['ema'].keys()}")
print(f"data2['model'].keys()={data2['model'].keys()}")

policy_old_1=data1['ema']
policy_old_2={
    key.replace('actor_old', 'network'): value for key, value in data2['model'].items() if 'actor_old' in key
}
print(f"policy_old_2.keys()={policy_old_2.keys()}")

def compare_checkpoints(dict1, dict2, rtol=1e-5, atol=1e-8):
    """
    Compare two checkpoint dictionaries for equality.
    rtol: relative tolerance for floating-point comparison
    atol: absolute tolerance for floating-point comparison
    Returns: True if checkpoints are identical (or close enough), False otherwise
    """
    tag=True
    # Check if dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        print("Dictionaries have different keys!")
        print(f"dict1 keys: {dict1.keys()}")
        print(f"dict2 keys: {dict2.keys()}")
        tag= False

    # Compare each key's value
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # Compare tensors
            if val1.shape != val2.shape:
                print(f"Key '{key}' tensors have different shapes: {val1.shape} vs {val2.shape}")
                tag= False
            if not torch.allclose(val1, val2, rtol=rtol, atol=atol):
                print(f"Key '{key}' tensors differ beyond tolerance!")
                max_diff = torch.max(torch.abs(val1 - val2)).item()
                print(f"Max difference: {max_diff}")
                tag= False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Recursively compare nested dictionaries (e.g., optimizer state_dict)
            if not compare_checkpoints(val1, val2, rtol, atol):
                print(f"Key '{key}' nested dictionaries differ!")
                tag= False
        else:
            # Compare non-tensor values (e.g., epoch number)
            if val1 != val2:
                print(f"Key '{key}' values differ: {val1} vs {val2}")
                tag= False

    return tag

# Compare the checkpoints
# are_equal12 = compare_checkpoints(data1, data2)
# print(f"Checkpoints are identical: {are_equal12}")
are_equal = compare_checkpoints(policy_old_1, policy_old_2)
print(f"Checkpoints are identical: {are_equal}")