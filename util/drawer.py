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
import matplotlib.pyplot as plt
import math

# Given parameters
clip_ploss_coef = 0.01
clip_ploss_coef_base = 0.001
clip_ploss_coef_rate = 3
ft_denoising_steps = 11  # To cover denoising_inds from 0 to 10

# Compute t and clip_ploss_coef
denoising_inds = np.arange(0, 11)  # 0 to 10 inclusive
t = denoising_inds / (ft_denoising_steps - 1)  # t ranges from 0 to 1
clip_ploss_coef_values = np.zeros_like(t, dtype=float)

for i, ti in enumerate(t):
    if ft_denoising_steps > 1:
        clip_ploss_coef_values[i] = clip_ploss_coef_base + (
            clip_ploss_coef - clip_ploss_coef_base
        ) * (np.exp(clip_ploss_coef_rate * ti) - 1) / (
            math.exp(clip_ploss_coef_rate) - 1
        )
    else:
        clip_ploss_coef_values[i] = ti

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(denoising_inds, clip_ploss_coef_values, marker='o', linestyle='-', color='b')
plt.xlabel('Denoising Index')
plt.ylabel('clip_ploss_coef')
plt.title('clip_ploss_coef vs. Denoising Index')
plt.grid(True)
plt.xticks(denoising_inds)
plt.savefig('./temp.png')