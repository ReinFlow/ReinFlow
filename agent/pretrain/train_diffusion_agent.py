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

import logging
log = logging.getLogger(__name__)
from agent.pretrain.train_agent import PreTrainAgent
from model.diffusion.diffusion import DiffusionModel



class TrainDiffusionAgent(PreTrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: DiffusionModel
        self.ema_model: DiffusionModel
        
        self.verbose_train=False #True #False
        self.verbose_loss= False #True #False 
        self.verbose_test= False #True #False
        
        if self.test_in_mujoco:
            self.test_denoising_steps=self.model.denoising_steps # 20
            self.test_model_type='ema' # 'original'        
            self.test_log_all = True
            self.only_test=False
            
        self.print_architecture()
    
    def print_architecture(self):
        import os
        arc_path=os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()
    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        loss_train = self.model.loss(*batch_data)
        return loss_train 

    def inference(self, cond:dict):
        '''for testing purpose'''
        if self.test_model_type == 'ema':
            samples = self.ema_model.forward(cond=cond, 
                                     deterministic=True)
        else:
            samples = self.model.forward(cond=cond, 
                                 deterministic=True)
        return samples