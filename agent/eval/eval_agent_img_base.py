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
Parent eval agent class with image inputs, for robomimic environment.
"""
import os
import logging
from tqdm import tqdm as tqdm
log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent  # Import the base class
import cv2

class EvalImgAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Image-specific initialization
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
    
    def single_run(self, num_denoising_steps, options_venv):
        
        self.video_writer = None
        if self.record_video:
            frame_width = 256  # Different from base class
            frame_height = 256  # Different from base class
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.eval_log_dir, f'{self.model.__class__.__name__}_{self.env_name}_step{num_denoising_steps}.mp4')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps, image"
        
        # Call the parent class's single_run with modified frame size
        result = super().single_run(num_denoising_steps, options_venv)
        return result