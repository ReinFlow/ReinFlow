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
Evaluate pre-trained/fine-tuned flow-matching policy.
"""
import logging
log = logging.getLogger(__name__)
from agent.eval.eval_agent_img_base import EvalImgAgent
from model.flow.reflow import ReFlow
from util.timer import Timer
# for robomimic
class EvalImgReFlowAgent(EvalImgAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.load_ema = cfg.get('load_ema', False) #Turn to True when evaluating pretrained models.
        self.clip_intermediate_actions=True
        self.denoising_steps =cfg.get("denoising_step_list", [1,2,4, 8,10, 16, 20,32,64,128])
        self.plot_scale='standard'
        self.render_onscreen = False
        self.record_video = False #True 
        self.record_env_index=0
        self.denoising_steps_trained = None # this is meaningless for reflow. it could be infinity. so we don't show the vertical line. 
        self.model.show_inference_process = False #True # whether to print each integration step during sampling. 
        ####################################################################################
        log.info(f"Evaluation: load_ema={self.load_ema}, clip_intermediate_actions={self.clip_intermediate_actions}")
    def infer(self, cond:dict, num_denoising_steps:int):
        ################################################      overload        #########################################################
        self.model: ReFlow
        timer = Timer()
        samples = self.model.sample(
                                    cond=cond, 
                                    inference_steps=num_denoising_steps, 
                                    record_intermediate=False, 
                                    clip_intermediate_actions=self.clip_intermediate_actions
                                    )
        duration = timer()
        # samples.trajectories: (n_envs, self.horizon_steps, self.action_dim)
        return samples, duration