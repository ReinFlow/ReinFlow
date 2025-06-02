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
fine-tuning.
"""
import os
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm as tqdm
import numpy as np
import torch
from agent.finetune.reinflow.train_ppo_agent import TrainPPOAgent
from model.flow.ft_ppo.pposhortcut import PPOShortCut
from agent.finetune.reinflow.buffer import PPOFlowBuffer#, PPOFlowBufferGPU
from util.scheduler_simple import get_scheduler
import matplotlib.pyplot as plt
from util.reproducibility import set_seed_everywhere
# define buffer on cpu or cuda. Currently GPU version is not offering significant acceleration...
# communication could be a bottleneck, it now just increases GPU volatile utilization from 7% to 13%
# this could own to mujoco generating data on cpu and we frequently moves them to and from GPUs. 

class TrainPPOShortCutAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Reward horizon --- always set to act_steps for now
        
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.inference_steps = self.model.inference_steps
        self.ft_denoising_steps = self.model.ft_denoising_steps
        self.repeat_samples = cfg.train.get("repeat_samples", False)
        
        self.normalize_act_space_dim = True   # normalize entropy and logprobability over horizon steps and action dimension. so that we don't need to adjust entropy coeff when env scales up. 
        self.normalize_denoising_horizon = True     # normalize denoising horizon when calculating the logprob of the markov chain of a single simulated action. 
        self.lr_schedule = cfg.train.lr_schedule
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", True)
        self.account_for_initial_stochasticity = cfg.train.get('account_for_initial_stochasticity', True)
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr
        self.critic_lr = cfg.train.critic_lr
        
        self.model: PPOShortCut
        self.model.seed=self.seed                       # for reproducibility.
        if self.model.noise_scheduler_type == 'const_schedule_itr':
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine_warmup',
                                                            min=0.016,
                                                            warmup_steps=self.n_train_itr * 0.01,
                                                            max=0.08, #0.15,
                                                            hold_steps=self.n_train_itr * 0.29,
                                                            anneal_steps=self.n_train_itr * 0.7)
            
            explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), explore_noises)
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise saved to %s" % name)
        elif self.model.noise_scheduler_type == 'learn_decay':
            max_std=cfg.model.max_logprob_denoising_std
            min_std=cfg.model.min_logprob_denoising_std
            self.max_noise_decay_ratio=cfg.train.get('max_noise_decay_ratio', 0.7)
            max_std_decayed=min_std*(1-self.max_noise_decay_ratio)+max_std*self.max_noise_decay_ratio
            # min_std*0.20+max_std*0.80 
            # min_std*0.4+max_std*0.6 
            # #cfg.model.min_logprob_denoising_std*1.5
            self.max_noise_hold_ratio=cfg.train.get('max_noise_hold_ratio', 0.35)
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine',
                                                            max=max_std,
                                                            hold_steps=self.n_train_itr * self.max_noise_hold_ratio,
                                                            anneal_steps=self.n_train_itr * (1-self.max_noise_hold_ratio),
                                                            min=max_std_decayed)
            max_explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f} to {max_std_decayed:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)
        else:
            max_std=cfg.model.max_logprob_denoising_std
            min_std=cfg.model.min_logprob_denoising_std
            max_explore_noises = [max_std for _ in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            log.info(f"Received self.model.noise_scheduler_type={self.model.noise_scheduler_type}, will use constant noise ranges [{min_std:.2f}, {max_std:.2f}]")
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name=os.path.join(self.logdir,'explore_noise')+'.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)
    
        self.initial_ratio_error_threshold = 1e-6 # for state based tasks, no augmentation, then the logprob ratio should be strictly 1.00 when batch=0 and epoch=0

    def init_buffer(self):
        self.buffer = PPOFlowBuffer(
            n_steps=self.n_steps,
            n_envs=self.n_envs,
            n_ft_denoising_steps= self.inference_steps, 
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
    
    def resume_training(self):
        super().resume_training()
        if self.model.noise_scheduler_type == 'const':
            updated_noise_std_range=[
                self.cfg.model.min_logprob_denoising_std, 
                self.cfg.model.max_logprob_denoising_std
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")   
    
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
                    action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(cond=cond, 
                                                                                          normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                                                          normalize_act_space_dimension=self.normalize_act_space_dim, 
                                                                                          clip_intermediate_actions=self.clip_intermediate_actions,
                                                                                          account_for_initial_stochasticity=self.account_for_initial_stochasticity) # for gpu version, add , device=self.device
                
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
                self.agent_update(verbose=self.verbose)
            
            # self.plot_state_trajecories() #(only in D3IL)
            
            self.log()                                          # diffusion_min_sampling_std
            self.update_lr()
            self.adjust_finetune_schedule()# update finetune scheduler of ReFlow Policy
            self.save_model()
            self.itr += 1 
            
    def adjust_finetune_schedule(self):
        # constant noise levels in intermediate steps, but the level changes over the course of training
        if self.model.noise_scheduler_type == 'const_schedule_itr':
            explore_noise_std = self.explore_noise_scheduler(self.itr)
            self.model.actor_ft.set_logprob_noise_levels(force_level=explore_noise_std)
        
        # gradually decrease the noise upper bound, to prevent noisy samples from hurting the model. 
        if self.model.noise_scheduler_type == 'learn_decay':
            updated_noise_std_range=[
                self.model.actor_ft.min_logprob_denoising_std, 
                self.explore_noise_scheduler(self.itr)
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")
        
    # overload...
    def save_model(self, only_save_policy_network=False):
        """
        saves model to disk; no ema recorded because we are doing RLFT.
        for evaluation purpose, set ``only_save_policy`` to True. This option does not save critic and exploration noise network and saves space. 
        for further training, set ``only_save_policy`` to False. This option saves everything needed to resume training. 
        """
        policy_network_state_dict = {
            'network.'+key:value for key, value in self.model.actor_ft.policy.state_dict().items()
        } # this is a FlowMLP network that can be loaded to the .network attribute of a ReFlow object. 
        
        if only_save_policy_network:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "policy": policy_network_state_dict,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        else:
            data = {
                "itr": self.itr,
                "cnt_train_steps": self.cnt_train_step,
                "model": self.model.state_dict(),  # for resume training
                "policy": policy_network_state_dict,  # flow policy for evaluation, without critic and exploration noise nets
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
                "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
            }
        
        # always save the last model for resume of training. 
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        
        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")
        
        # save the best model evaluated so far 
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir,f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far =False
    
    @torch.no_grad()
    def get_samples_logprobs(self, 
                             cond:dict, 
                             ret_device='cpu', 
                             save_chains=True, 
                             normalize_denoising_horizon=False, 
                             normalize_act_space_dimension=False, 
                             clip_intermediate_actions=True,
                             account_for_initial_stochasticity=True):
        # returns: action_samples are still numpy because mujoco engine receives np.
        # set_seed_everywhere(self.seed)
        if save_chains:
            action_samples, chains_venv, logprob_venv  = self.model.get_actions(cond, 
                                                                                eval_mode=self.eval_mode, 
                                                                                save_chains=save_chains, 
                                                                                normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                                clip_intermediate_actions=clip_intermediate_actions,
                                                                                account_for_initial_stochasticity=account_for_initial_stochasticity)        # n_envs , horizon_steps , act_dim
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv, logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
        else:
            action_samples, logprob_venv  = self.model.get_actions(cond, 
                                                                   eval_mode=self.eval_mode, 
                                                                   save_chains=save_chains, 
                                                                   normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                   normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                   clip_intermediate_actions=clip_intermediate_actions,
                                                                   account_for_initial_stochasticity=account_for_initial_stochasticity)
            return action_samples.cpu().numpy(), logprob_venv.cpu().numpy()  if ret_device=='cpu' else logprob_venv
    
    def get_value(self, cond:dict, device='cpu'):
        # cond contains a floating-point torch.tensor on self.device
        if device == 'cpu':
            value_venv = self.model.critic.forward(cond).cpu().numpy().flatten()
        else:
            value_venv = self.model.critic.forward(cond).squeeze().float().to(self.device)
        return value_venv
    
    # overload
    def update_lr(self, val_metric=None):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':   # adapt learning rate according to kl divergence on each minibatch.
            return
        else: # use predefined lr scheduler. 
            super().update_lr()
    
    def update_lr_adaptive_kl(self, approx_kl):
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        min_critic_lr = 1e-5
        max_critic_lr = 1e-3
        tune='maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            self.critic_lr = max(min_critic_lr, self.critic_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            self.critic_lr = min(max_critic_lr, self.critic_lr * 1.5)
            tune = 'increases'
        for actor_param_group, critic_param_group in zip(self.actor_optimizer.param_groups, self.critic_optimizer.param_groups):
            actor_param_group["lr"] = self.actor_lr
            critic_param_group["lr"] = self.critic_lr
        log.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    def minibatch_generator(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        self.total_steps = self.n_steps * self.n_envs
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    returns[inds_b], 
                    oldvalues[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch    

    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        
        duplicate_multiplier = 10   #self.ft_denoising_steps
        
        self.total_steps = self.n_steps * self.n_envs *  duplicate_multiplier
        
        for update_epoch in range(self.update_epochs):
            self.kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            if self.lr_schedule=='fixed' and self.kl_change_too_much:
                break
            for batch_id, start in enumerate(range(0, self.total_steps, self.batch_size)):
                end = start + self.batch_size
                inds_b = indices[start:end]
                batch_inds_b, denoising_inds_b = torch.unravel_index(
                    inds_b,
                    (self.n_steps * self.n_envs, duplicate_multiplier),
                )
                minibatch = (
                    {"state": obs[batch_inds_b]},
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                if self.lr_schedule=='fixed' and self.target_kl and self.approx_kl > self.target_kl: # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch

    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():

            # minibatch gradient descent
            self.model: PPOShortCut
            
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, Q_values= self.model.loss(*minibatch, 
                                                    use_bc_loss=self.use_bc_loss, 
                                                    bc_loss_type=self.bc_loss_type, normalize_denoising_horizon=self.normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            # update policy and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            loss.backward()
            
            # debug the losses
            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
            actor_old_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_old.parameters(), max_norm=float('inf'))
            critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}, actor_old_norm={actor_old_norm:.2e}")
            
            # always and frequently update critic
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            
            # after critic warmup to make the value estimate a reasonable value, update the actor less frequently but more times. 
            if self.itr >= self.n_critic_warmup_itr:
                if (self.itr-self.n_critic_warmup_itr) % self.actor_update_freq ==0:
                    for _ in range(self.actor_update_epoch):
                        if self.max_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                        self.actor_optimizer.step()
        
        clip_fracs=np.mean(clipfracs_list)
        noise_stds=np.mean(noise_std_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "bc_loss": bc_loss,
                "approx kl": self.approx_kl,
                "ratio": ratio,
                "clipfrac": clip_fracs,
                "explained variance": self.explained_var,
                "old_logprob_min": oldlogprob_min,
                "old_logprob_max": oldlogprob_max,
                "old_logprob_std": oldlogprob_std,
                "new_logprob_min": newlogprob_min,
                "new_logprob_max": newlogprob_max,
                "new_logprob_std": newlogprob_std,
                "actor_norm": actor_norm,
                "critic_norm": critic_norm,
                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                "min_logprob_noise_std": self.model.min_logprob_denoising_std,
                "min_sampling_noise_std": self.model.min_sampling_denoising_std,
                "noise_std": noise_stds,
                "Q_values": Q_values
            }
    
    