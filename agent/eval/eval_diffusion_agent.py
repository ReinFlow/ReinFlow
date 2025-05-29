# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tqdm import tqdm as tqdm
import logging
log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent
from model.diffusion.diffusion import DiffusionModel
from util.timer import Timer

# Save the figure
class EvalDiffusionAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.load_ema = cfg.get('load_ema', False) # Turn to True when evaluating pretrained models.
        self.record_video =False # False
        self.record_env_index=0
        self.render_onscreen =False #True
        self.denoising_steps = cfg.get("denoising_step_list", [1,2, 4,8,10,12,14,16,18,20,22,24,32,64,128,256])
        self.denoising_steps_trained = self.cfg.denoising_steps
        self.plot_scale='standard'
        self.ddim_eta = 1.0   # eta==1: DDPM   eta==0: DDIM
        ################################################################################################################################
    def infer(self,cond:dict, num_denoising_steps:int):
        ################################################      overload        #########################################################
        self.model: DiffusionModel # DDPM or DDIM
        if self.model.use_ddim:
            self.model.ddim_steps = num_denoising_steps
            self.model.ddim_eta = self.ddim_eta
        else:
            self.model.denoising_steps = num_denoising_steps
        self.model.calculate_parameters() # reset all the ddpm or ddim parameters based on number of generation steps.
        
        timer = Timer()
        samples = self.model.forward(cond=cond, deterministic=True)
        duration = timer()
        
        return samples, duration
        ################################################################################################################################
        