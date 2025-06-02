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
import os
from util.process import sort_handles_by_labels
# Data
episodes = np.array([16, 32, 64, 100])
pretrained_success = np.array([3.075, 10.150, 27.670, 25.140])/100.00
finetuned_success = np.array([0.000, 20.900, 61.300, 77.400])/100.00
finetuned_std = np.array([0.000, 1.699, 3.850, 2.118])/100.00

# Plot settings
finetuned_color = 'purple'
pretrained_color = 'plum'
font_size = 24
log_dir = 'visualize/Finetune_Data_Scale_v.s._Reward_square'  # Current directory; change as needed
model_name = 'Shortcut'
env_name = 'Square'

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot lines with semilogx scale (base 2)
ax.set_xscale('log', base=2)
handle_pretrained, =ax.semilogx(episodes, pretrained_success, color=pretrained_color, lw=10, base=2)
handle_finetuned=ax.errorbar(episodes, finetuned_success, yerr=finetuned_std, color=finetuned_color, lw=6, capsize=6, capthick=4, fmt='-')

handles_sorted, labels_sorted=sort_handles_by_labels([handle_pretrained, handle_finetuned], ['Pre-trained', 'Fine-tuned'], ['Fine-tuned', 'Pre-trained'])

# Customize axes
ax.set_xticks([16, 32, 64, 128])
ax.set_xticklabels(['16', '32', '64', '128'], fontsize=font_size)
ax.set_xlabel('Pretrained Episodes', fontsize=font_size, labelpad=10)
ax.set_ylabel('Success Rate', fontsize=font_size, labelpad=10)
# ax.set_title(f'Success Rate of {model_name} in {env_name}', fontsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)

# Ensure y-axis tick labels are integers
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

# Add legend
ax.legend(handles_sorted, labels_sorted, fontsize=28, loc='upper left')

# Adjust layout
fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

# Save figure
fig_path_png = os.path.join(log_dir, 'datascale_shortcut_square_success_rate_2d.png')
fig_path_pdf = os.path.join(log_dir, 'datascale_shortcut_square_success_rate_2d.pdf')
plt.savefig(fig_path_png, bbox_inches='tight', dpi=300)
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"2D plot saved to {fig_path_png} and {fig_path_pdf}")
plt.show()
plt.close()

