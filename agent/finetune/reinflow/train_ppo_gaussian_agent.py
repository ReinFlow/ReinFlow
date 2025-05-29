"""
PPO training for Gaussian/GMM policy.
Run this line to train a gaussian policy from scratch: 

python script/run.py --config-dir=cfg/gym/scratch/hopper-v2 --config-name=ppo_gaussian_mlp device=cuda:6 wandb=null
"""
import torch
import logging
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from agent.finetune.reinflow.buffer import PPOBuffer
from typing import Tuple
import numpy as np
from model.gaussian.gaussian_ppo import PPO_Gaussian

class TrainPPOGaussianAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.buffer = PPOBuffer(n_steps=self.n_steps, 
                                      n_envs=self.n_envs,
                                      horizon_steps=self.horizon_steps, 
                                      act_steps= self.act_steps,
                                      action_dim=self.action_dim,
                                      n_cond_step=self.n_cond_step, 
                                      obs_dim=self.obs_dim, 
                                      save_full_observation=self.save_full_observations,
                                      furniture_sparse_reward = self.furniture_sparse_reward,
                                      best_reward_threshold_for_success = self.best_reward_threshold_for_success,
                                      reward_scale_running = self.reward_scale_running,
                                      gamma = self.gamma,
                                      gae_lambda=self.gae_lambda,
                                      reward_scale_const = self.reward_scale_const,
                                      device=self.device)
    
    def run(self):
        self.model: PPO_Gaussian
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), it means the buffer automatically resets when new itr begins (step =0) so we only need to reset in the beginning. This works only for PPO buffer.
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env()
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                with torch.no_grad():
                    value_venv = self.model.critic.forward(torch.tensor(self.prev_obs_venv["state"]).float().to(self.device)).cpu().numpy().flatten()
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    action_samples, logprob_venv = self.get_samples_logprobs(cond=cond)
                    
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                # save to buffer
                self.buffer.add(step, self.prev_obs_venv["state"], action_samples, reward_venv,terminated_venv, truncated_venv, value_venv, logprob_venv)
                self.buffer.save_full_obs(info_venv)
                
                # update for next step
                self.prev_obs_venv = obs_venv
                self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0 #not acounting for done within action chunk
            
            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic)
                self.agent_update()
            
            self.plot_state_trajecories() #(only in D3IL)

            self.update_lr()
            
            self.save_model()
            
            self.log()
            
            self.itr += 1
            
    @torch.no_grad
    def get_samples_logprobs(self, cond:dict)->Tuple[np.ndarray, np.ndarray]:
        '''
        input:
            cond: dict, which contains conda["state"]: torch.Tensor(self.n_envs, self.n_cond_steps, self.obs_dim)
        output:
            action_samples: self.n_envs x self.horizon_steps x self.act_dim
            logprob_venv:   self.n_envs x self.horizon_steps
        in order to accomodate mujoco simulator, we return numpy ndarray on cpu. so output are numpy ndarrays
        '''
        action_samples = self.model.forward(
                            cond=cond,
                            deterministic=self.eval_mode,
                        )
        logprob_venv, _, _ = self.model.get_logprobs(cond, action_samples)
        
        return action_samples.cpu().numpy(), logprob_venv.cpu().numpy()  
    
    
    def update_step(self, batch:tuple):
        '''
        input: 
            batch:, a tuple, containing:
                {"state": obs[minibatch_idx]},  : dict, 
                samples[minibatch_idx]          : torch.Tensor(minibatch_size, horizo_steps, act_dim)
                returns[minibatch_idx],
                values[minibatch_idx],
                advantages[minibatch_idx],
                logprobs[minibatch_idx]]
        output:
            pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std: 
            float, float, float, float, float, float, float, float
        '''
        
        pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.model.loss(*batch, use_bc_loss=self.use_bc_loss)
        
        return pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std