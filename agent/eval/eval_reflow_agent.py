"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.
self.model: Flow
"""
import logging
log = logging.getLogger(__name__)
from agent.eval.eval_agent_base import EvalAgent
from model.flow.reflow import ReFlow
from util.timer import Timer

class EvalReFlowAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        ################################################      overload        #########################################################
        self.load_ema = cfg.get('load_ema', False) #Turn to True when evaluating pretrained models.
        self.clip_intermediate_actions=True
        self.record_video =False
        self.record_env_index=0
        self.render_onscreen =False #not self.record_video #False
        self.denoising_steps = cfg.get("denoising_step_list", [1,2, 4,8,10,12,14,16,18,20,22,24,32,64,128,256])
        self.denoising_steps_trained = None # actually this is meaning less for reflow. it could be infinity. 
        self.model.show_inference_process = False # whether to print each integration step during sampling. 
        self.plot_scale='standard'
        log.info(f"Evaluation: load_ema={self.load_ema}, clip_intermediate_actions={self.clip_intermediate_actions}")
        ####################################################################################
    def infer(self, cond:dict, num_denoising_steps:int):
        ################################################      overload        #########################################################
        self.model: ReFlow
        timer = Timer()
        samples = self.model.sample(cond=cond, 
                                    inference_steps=num_denoising_steps, 
                                    record_intermediate=False, 
                                    clip_intermediate_actions=self.clip_intermediate_actions)# samples.trajectories: (n_envs, self.horizon_steps, self.action_dim)
        duration = timer()
        return samples, duration
    
    