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
# 
# 

"""
Pre-training ReFlow policy
"""
import logging
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.flow.reflow import ReFlow
class TrainReFlowAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: ReFlow
        self.ema_model: ReFlow
        
        self.verbose_train=False #True #False #True #False #True #False
        self.verbose_loss= True #False #False #True #False #True #False # True
        self.verbose_test= False #True #False
        
        if self.test_in_mujoco:
            self.test_log_all = True
            self.only_test=  False                      # when toggled, test and then exit code.
            
            self.test_denoising_steps=4                 #self.model.max_denoising_steps!!!!!!!!!!!!
            
            self.test_clip_intermediate_actions=True    #True    # this will affect performance. !!!!!!!!!!!!
            
            self.test_model_type='ema'                  # minor difference!!!!!!!!!!!!
    
    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        # here *batch_data = actions, observation, according to StitchedSequenceDataset
        act, cond = batch_data
        (xt, t), v = self.model.generate_target(act)  # here *batch_train = actions, observation, according to StitchedSequenceDataset
        loss= self.model.loss(xt, t, cond, v)
        return loss
    
    def inference(self, cond:dict):
        '''for testing purpose'''
        if self.test_model_type == 'ema':
            samples = self.ema_model.sample(cond, 
                                            inference_steps=self.test_denoising_steps, 
                                            record_intermediate=False,
                                            clip_intermediate_actions=self.test_clip_intermediate_actions)
        else:
            samples = self.model.sample(cond, 
                                        inference_steps=self.test_denoising_steps, 
                                        record_intermediate=False,
                                        clip_intermediate_actions=self.test_clip_intermediate_actions)
        return samples 
    
    
            
            