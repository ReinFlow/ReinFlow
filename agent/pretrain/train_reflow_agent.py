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
    
    
            
            