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
DPPO fine-tuning.
run this line to finetune hopper-v2: 
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_diffusion_mlp device=cuda:7 wandb=null
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_diffusion_mlp device=cuda:6 wandb=null
"""
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from util.scheduler import CosineAnnealingWarmupRestarts
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from model.diffusion.diffusion_ppo import PPODiffusion
from agent.finetune.reinflow.buffer import PPODiffusionBuffer, PPODiffusionBufferGPU 
# define buffer on cpu or cuda. Currently GPU version is not offering significant acceleration...communication could be a bottleneck. It just increases GPU volatile utilization from 7% to 13% 
# this is partially due to mujoco engine is still on cpu

class TrainPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.ft_denoising_steps = self.model.ft_denoising_steps
        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta

        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        self.model: PPODiffusion
        
        
        
        
        
    def init_buffer(self):
        self.buffer = PPODiffusionBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps= self.ft_denoising_steps, 
            horizon_steps=self.horizon_steps,
            act_steps=self.act_steps,
            action_dim=self.action_dim,
            n_cond_step=self.n_cond_step,
            obs_dim=self.obs_dim,
            save_full_observation=self.save_full_observations,
            furniture_sparse_reward=self.furniture_sparse_reward,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            reward_scale_running=self.reward_scale_running,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reward_scale_const=self.reward_scale_const,
            device=self.device,
        )
        
    def adjust_finetune_schedule(self):
        self.model.step()
        self.diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()
    
    def get_samples_logprobs(self, cond:dict, device='cpu'):
        samples = self.model.forward(cond=cond, deterministic=self.eval_mode, return_chain=True)
        action_samples = samples.trajectories                               # n_envs , horizon_steps , act_dim
        chains_venv = samples.chains                                        # n_envs , ft_denoising_steps+1 , horizon_steps , act_dim = (40, 11, 4, 3)
        logprob_venv = self.model.get_logprobs(cond, chains_venv).reshape(self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim)
        # get_logprobs returns [(n_envs x ft_denoising_steps) , horizon_steps , act_dim], but we convert it to [n_envs, ft_denoising_steps , horizon_steps , act_dim]
        
        if device=='cpu':
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy(), logprob_venv.cpu().numpy()
        else:
            return action_samples.cpu().numpy(), chains_venv, logprob_venv  # action_samples are still numpy because mujoco engine receives np.
    
    def get_value(self, cond:dict, device='cpu'):
        # cond contains a floating-point torch.tensor on self.device
        if device == 'cpu':
            value_venv = self.model.critic.forward(cond).cpu().numpy().flatten()
        else:
            value_venv = self.model.critic.forward(cond).squeeze().float().to(self.device)
        return value_venv
                    
    def agent_update(self, verbose=False):
        obs, chains, returns, oldvalues, advantages, oldlogprobs = self.buffer.make_dataset()
        Q_values=0.0 
        if verbose:
            log.info(f"obs shape: {obs.shape}")                 # obs shape: torch.Size([20000, 1, 11])
            log.info(f"chains shape: {chains.shape}")           # torch.Size([20000, 11, 4, 3])
            log.info(f"returns shape: {returns.shape}")         # torch.Size([20000])
            log.info(f"values shape: {oldvalues.shape}")           # torch.Size([20000])
            log.info(f"advantages shape: {advantages.shape}")   # torch.Size([20000])
            log.info(f"logprobs shape: {oldlogprobs.shape}")       # torch.Size([20000, 10, 4, 3])
        
        
        # Explained variation of future rewards using value function
        explained_var = self.buffer.get_explained_var(oldvalues, returns)
        Q_values = oldvalues.mean().item() # appended by ReinFlow Authors
        # Update policy and critic
        clipfracs_list = []
        self.total_steps = self.n_steps * self.n_envs * self.ft_denoising_steps   # overload. 500 x 40 x 10 = 20_000 x 10 = 200_000 for hopper. 
        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]                            # b is for batch. self.batch_size = 50_000 
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, self.ft_denoising_steps),
                )
                if verbose:
                    log.info(f"batch_inds_b={batch_inds_b.shape}, denoising_inds_b={denoising_inds_b.shape}")
                minibatch = (
                    {"state": obs[batch_inds_b]},                      # obs_b
                    chains[batch_inds_b, denoising_inds_b],            # chains_prev_b    self.batch_size x self.horizon_steps x self.act_dim == [B, 4, 3] for hopper
                    chains[batch_inds_b, denoising_inds_b + 1],        # chains_next_b    self.batch_size x self.horizon_steps x self.act_dim == [B, 4, 3] B=50_000
                    denoising_inds_b,                                  # denoising_inds_b [B,] = [50_000,] 
                    returns[batch_inds_b],                             # returns_b        [B,] = [50_000,] 
                    oldvalues[batch_inds_b],                              # values_b         [B,] = [50_000,] 
                    advantages[batch_inds_b],                          # advantages_b     [B,] = [50_000,]  there are many duplicated entries.
                    oldlogprobs[batch_inds_b, denoising_inds_b]           # logprobs_b       self.batch_size x self.ft_denoising_step x self.horizon_steps x self.act_dim == [50_000, 10, 4, 3]
                )
                if verbose:
                    log.info(f"returns[batch_inds_b]={returns[batch_inds_b].shape}, values[batch_inds_b]={oldvalues[batch_inds_b].shape}, \
                        advantages[batch_inds_b]={advantages[batch_inds_b].shape},\
                            logprobs[batch_inds_b]={oldlogprobs[batch_inds_b].shape}")
                
                # minibatch gradient descent
                self.model: PPODiffusion
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, eta = self.model.loss(*minibatch, 
                                                                                                          use_bc_loss=self.use_bc_loss, 
                                                                                                          reward_horizon=self.reward_horizon)
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff

                clipfracs_list += [clipfrac]
                
                # update policy and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if self.learn_eta:
                    self.eta_optimizer.zero_grad()
                
                loss.backward()
                
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor_ft.parameters(), self.max_grad_norm
                        )
                    self.actor_optimizer.step()
                    if self.learn_eta and batch_id % self.eta_update_interval == 0:
                        self.eta_optimizer.step()
                self.critic_optimizer.step()
                
                if verbose: 
                    log.info(f"update_epoch: {update_epoch}, num_batch: {max(1, self.total_steps // self.batch_size)}, approx_kl: {approx_kl}")
                
                if self.target_kl is not None and approx_kl > self.target_kl:
                    kl_change_too_much = True
                    break
            if kl_change_too_much:
                break
        clip_fracs=np.mean(clipfracs_list)
        
        self.train_ret_dict = {
                    "loss": loss,
                    "pg loss": pg_loss,
                    "value loss": v_loss,
                    "entropy_loss": entropy_loss,
                    "bc_loss": bc_loss,
                    "approx kl": approx_kl,
                    "ratio": ratio,
                    "clipfrac": clip_fracs,
                    "explained variance": explained_var,
                    "eta": eta,
                    "Q_values": Q_values
                }

    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env() # for gpu version, add device=self.device
            self.buffer.update_full_obs()
            for step in range(self.n_steps):
                with torch.no_grad():
                    cond = {
                        "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                    }
                    value_venv = self.get_value(cond=cond) # for gpu version add , device=self.device
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(cond=cond) # for gpu version, add , device=self.device
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                self.buffer.save_full_obs(info_venv)
                self.buffer.add(step, self.prev_obs_venv["state"], chains_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            self.buffer.summarize_episode_reward()
            
            if not self.eval_mode:
                self.buffer.update(obs_venv, self.model.critic) # for gpu version, add device=self.device
                self.agent_update() 
            
            self.plot_state_trajecories() #(only in D3IL)
            
            self.adjust_finetune_schedule()
            self.log(train_prt_str_additional=f"diffusion - min sampling std:{self.diffusion_min_sampling_std}")
            
            self.update_lr()
            if self.itr >= self.n_critic_warmup_itr and self.learn_eta:
                self.eta_lr_scheduler.step()
            # update finetune scheduler of Diffusion Policy
            self.save_model()
            
            self.itr += 1