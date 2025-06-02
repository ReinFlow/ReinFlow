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
from utils import merge_eval_statistics

if __name__=="__main__":
    MODEL_NAME='DDIM'
    ENV_NAME='can'
    path_1='visualize/0_Models/DiffusionModel/can/25-04-16-20-33-20/eval_statistics.npz'
    #'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-00-31-10-DDPM/eval_statistics.npz'
    path_2='visualize/0_Models/DiffusionModel/can/25-04-17-11-13-29/eval_statistics.npz'
    #'visualize/0_Models/DiffusionModel/walker2d-medium-v2/25-04-16-13-45-24-DDPM2/eval_statistics.npz'
    out_dir=f'visualize/{MODEL_NAME}/{ENV_NAME}/25-04-17-21-12-00-DDIM-can/'
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, 'eval_statistics.npz')
    merged_stats=merge_eval_statistics(
        path_1,
        path_2,
        out_path
    )
    print(f"merged_stats={merged_stats}")