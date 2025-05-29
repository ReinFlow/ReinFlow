"""
DPPO fine-tuning.
run this line to finetune hopper-v2: 
python script/run.py --config-dir=cfg/gym/finetune/hopper-v2 --config-name=ft_ppo_reflow_mlp device=cuda:7
"""
from tqdm import tqdm as tqdm
import torch
import logging
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from model.common.modules import RandomShiftsAug
import numpy as np
from model.flow.ft_ppo.ppoflow import PPOFlow
from agent.finetune.reinflow.buffer import PPOFlowImgBuffer, PPOFlowImgBufferGPU

class TrainPPOImgFlowAgent(TrainPPOFlowAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)
        
        self.initial_ratio_error_threshold = 1e-6 # for image input tasks with random augmentation, the log prob at eopch=0 batch=0 could be different. so we relax the threshold.
        
        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate
        
        self.verbose = cfg.train.get('verbose', False)
        
        self.buffer_device = self.device # 'cpu'

        self.minibatch_duplicate_multiplier= 5    #self.ft_denoising_steps for robomimic

        self.skip_initial_eval =False
        
        self.use_early_stop = True
        
        self.fix_nextvalue_augment_bug=True #False
        
    # overload
    def init_buffer(self):
        log.info(f"self.buffer_device={self.buffer_device}")
        log_prob_cfg_dict={
            'normalize_denoising_horizon':self.normalize_denoising_horizon,
            'normalize_act_space_dimension': self.normalize_act_space_dim, 
            'clip_intermediate_actions':self.clip_intermediate_actions,
            'account_for_initial_stochasticity':self.account_for_initial_stochasticity
        }       
                                                           
        if self.buffer_device=='cpu':
            self.buffer = PPOFlowImgBuffer(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.inference_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims,  # this is different from state obs
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug,       # debug bugfix 
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug, # debug bugfix 2
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        else:
            self.buffer = PPOFlowImgBufferGPU(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.inference_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims,  # this is different from state obs
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug,       # debug bugfix 
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug, # debug bugfix 2
                device=self.device,
                log_prob_cfg_dict=log_prob_cfg_dict
            )
        log.info(f"created buffer: {self.buffer.__class__} on {self.buffer_device}")

    # overload, do not return log probabilities during sampling.
    @torch.no_grad()
    def get_samples(self, 
                    cond:dict, 
                    ret_device='cpu', 
                    save_chains=True, 
                    normalize_denoising_horizon=False, 
                    normalize_act_space_dimension=False, 
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True):
        # returns: action_samples are still numpy because mujoco engine receives np.
        if save_chains:
            action_samples, chains_venv  = self.model.get_actions(cond, 
                                                                eval_mode=self.eval_mode, 
                                                                save_chains=save_chains, 
                                                                normalize_denoising_horizon=normalize_denoising_horizon, 
                                                                normalize_act_space_dimension=normalize_act_space_dimension, 
                                                                clip_intermediate_actions=clip_intermediate_actions,
                                                                account_for_initial_stochasticity=account_for_initial_stochasticity,
                                                                ret_logprob=False)        # n_envs , horizon_steps , act_dim
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy() if ret_device=='cpu' else chains_venv
        else:
            action_samples  = self.model.get_actions(cond, 
                                                    eval_mode=self.eval_mode, 
                                                    save_chains=save_chains, 
                                                    normalize_denoising_horizon=normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=normalize_act_space_dimension, 
                                                    clip_intermediate_actions=clip_intermediate_actions,
                                                    account_for_initial_stochasticity=account_for_initial_stochasticity,
                                                    ret_logprob=False)
            return action_samples.cpu().numpy()
    
    
    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # as long as we put items at the right position in the buffer (determined by 'step'), the buffer automatically resets when new iteration begins (step =0). so we only need to reset in the beginning. This works only for PPO buffer, otherwise may need to reset when new iter begins.
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.buffer.update_full_obs()
            for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                if not self.verbose and step % 100 == 0: print(f"Processed {step} of {self.n_steps}")
                with torch.no_grad():
                    ####### visual input #########################
                    cond = {
                        key: torch.from_numpy(self.prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }
                    ## overload bug fix
                    action_samples, chains_venv = self.get_samples(cond=cond, 
                                                                   ret_device=self.buffer_device,
                                                                    normalize_denoising_horizon=self.normalize_denoising_horizon,
                                                                    normalize_act_space_dimension=self.normalize_act_space_dim, 
                                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
                
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                
                # overload, bug fix
                self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, terminated_venv, truncated_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            
            self.buffer.summarize_episode_reward()
            if not self.eval_mode:
                ### bug fix
                self.buffer: PPOFlowImgBufferGPU
                self.buffer.update_img(obs_venv, self.model)
                self.agent_update(verbose=self.verbose)
            
            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()# update finetune scheduler of ReFlow Policy
            self.save_model()
                              
            
            self.itr += 1
            # early stopping
            if self.use_early_stop and (self.buffer.success_rate < 0.05 or self.buffer.avg_episode_reward < 2.0):
                log.info(f"Your finetuning failed. success_rate={self.buffer.success_rate*100:.2f}% and avg_episode_reward={self.buffer.avg_episode_reward:.2f}")
                exit()
            
            self.clear_cache()
            self.inspect_memory()
            
    # overload to accomodate gradaccum
    def agent_update(self, verbose=True):
        clipfracs_list = []
        noise_std_list = []
        actor_norm=0.0
        critic_norm=0.0
        
        for update_epoch, batch_id, minibatch in self.minibatch_generator() if not self.repeat_samples else self.minibatch_generator_repeat():

            # minibatch gradient descent
            self.model: PPOFlow
            
            pg_loss, entropy_loss, v_loss, bc_loss, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
                newlogprob_min, newlogprob_max, newlogprob_std, \
                noise_std, newQ_values= self.model.loss(*minibatch, 
                                                    use_bc_loss=self.use_bc_loss, 
                                                    bc_loss_type=self.bc_loss_type, 
                                                    normalize_denoising_horizon=self.normalize_denoising_horizon, 
                                                    normalize_act_space_dimension=self.normalize_act_space_dim,
                                                    verbose=verbose,
                                                    clip_intermediate_actions=self.clip_intermediate_actions,
                                                    account_for_initial_stochasticity=self.account_for_initial_stochasticity)
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}/{max(1, self.total_steps // self.batch_size)}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")
            
            if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> self.initial_ratio_error_threshold:
                log.info(f"Warning: ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
            
            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)
            
            loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
            
            clipfracs_list += [clipfrac]
            noise_std_list += [noise_std]
            
            loss.backward()
            # overload, bugfix to support gradient accumulation
            if (batch_id + 1) % self.grad_accumulate == 0:
                # debug the losses
                actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
                critic_norm = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), max_norm=float('inf'))
                if verbose:
                    log.info(f"before clipping: actor_norm={actor_norm:.2e}, critic_norm={critic_norm:.2e}")
                
                # update actor: after critic warmup update the actor less frequently but more times. 
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                # update critic
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # release gradient accumulation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # report
                log.info(f"run grad update at batch {batch_id}")
                log.info(
                    f"approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps //self.batch_size}"
                )
        
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
                "Q_values": self.Q_values   # # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values but also remember to re-define Q values in agent/finetune/reinflow/train_ppo_diffusion_img_agent.py
            }
    
    def minibatch_generator_repeat(self):
        self.approx_kl = 0.0
        
        obs, chains, returns, oldvalues, advantages, oldlogprobs =  self.buffer.make_dataset()
        # Explained variation of future rewards using value function
        self.explained_var = self.buffer.get_explained_var(oldvalues, returns)
        # define Q values as the old Q values to align with the definition in diffusion ppo. you can change those back to new Q values but also remember to re-define Q values in agent/finetune/reinflow/train_ppo_diffusion_img_agent.py
        self.Q_values = oldvalues.mean().item()
        
        duplicate_multiplier = self.minibatch_duplicate_multiplier 
        
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
                    {k: obs[k][batch_inds_b] for k in obs}, # rgb and state. overload
                    chains[batch_inds_b],
                    returns[batch_inds_b], 
                    oldvalues[batch_inds_b],
                    advantages[batch_inds_b],
                    oldlogprobs[batch_inds_b] 
                )
                
                if (self.lr_schedule=='fixed' 
                    and self.target_kl 
                    and self.approx_kl > self.target_kl
                    and self.itr >= self.n_critic_warmup_itr  # bug fix
                ): # we can also use adaptive KL instead of early stopping.
                    self.kl_change_too_much = True
                    log.warning(f"KL change too much, approx_kl ={self.approx_kl} > {self.target_kl} = target_kl, stop optimization.")
                    break
                
                yield update_epoch, batch_id, minibatch
                
        
        