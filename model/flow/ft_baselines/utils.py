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


import numpy as np
def td_values(states, rewards, terminateds, state_values, gamma=0.99, alpha=0.95, lam=0.95):
    """
    Compute TD(λ) estimates for a list of samples. 
    This snippet is taken from agent/finetune/diffusion_baselines/train_awr_diffusion_agent.py
    
    Args:
        states: List of state observations (np.ndarray).
        rewards: List of rewards (np.ndarray).
        terminateds: List of termination flags (np.ndarray).
        state_values: Estimated state values (np.ndarray).
        gamma: Discount factor (float).
        alpha: TD learning rate (float).
        lam: Lambda for TD(λ) (float).

    Returns:
        np.ndarray: TD(λ) estimates.
    """
    sample_count = len(states)
    tds = np.zeros_like(state_values, dtype=np.float32)
    next_value = state_values[-1].copy()
    next_value[terminateds[-1]] = 0.0

    val = 0.0
    for i in range(sample_count - 1, -1, -1):
        if i < sample_count - 1:
            next_value = state_values[i + 1]
            next_value = next_value * (1 - terminateds[i])
        state_value = state_values[i]
        error = rewards[i] + gamma * next_value - state_value
        val = alpha * error + gamma * lam * (1 - terminateds[i]) * val
        tds[i] = val + state_value
    return tds

    
    