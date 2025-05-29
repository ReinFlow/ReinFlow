"""
DPPO fine-tuning for pixel observations.
We obtain action and chains from raw pixels
We obtain values, advantages, log probabilities from augmented pixels.
"""
from tqdm import tqdm as tqdm
import numpy as np
import torch
import logging
log = logging.getLogger(__name__)
from model.diffusion.diffusion_ppo import PPODiffusion
from agent.finetune.reinflow.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.common.modules import RandomShiftsAug
from agent.finetune.reinflow.buffer import PPODiffusionImgBuffer, PPODiffusionImgBufferGPU

class TrainPPOImgDiffusionAgent(TrainPPODiffusionAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}
        
        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate
        
        self.verbose = cfg.train.get('verbose', False)

        self.buffer_device = self.device # 'cpu' #
        self.buffer: PPODiffusionImgBufferGPU
        
        self.skip_initial_eval = cfg.train.get('skip_initial_eval', False)
        
        self.fix_nextvalue_augment_bug=False
        
    # overload
    def init_buffer(self):
        log.info(f"self.buffer_device={self.buffer_device}")
        if self.buffer_device=='cpu':
            self.buffer= PPODiffusionImgBuffer(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.ft_denoising_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims, # rgb state
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug if self.augment else None,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device, 
            )
        else:
            self.buffer= PPODiffusionImgBufferGPU(
                n_steps=self.n_steps,
                n_envs=self.n_envs,
                n_ft_denoising_steps= self.ft_denoising_steps, 
                horizon_steps=self.horizon_steps,
                act_steps=self.act_steps,
                action_dim=self.action_dim,
                n_cond_step=self.n_cond_step,
                obs_dim=self.obs_dims, # rgb state
                save_full_observation=self.save_full_observations,
                furniture_sparse_reward=self.furniture_sparse_reward,
                best_reward_threshold_for_success=self.best_reward_threshold_for_success,
                reward_scale_running=self.reward_scale_running,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                reward_scale_const=self.reward_scale_const,
                aug=self.aug if self.augment else None,
                fix_nextvalue_augment_bug=self.fix_nextvalue_augment_bug,
                device=self.device, 
            )
        log.info(f"created buffer: {self.buffer.__class__}")
    
    # overload 
    def agent_update(self, verbose=False):
        Q_values = 0.0
        obs, chains, returns, oldvalues, advantages, oldlogprobs = self.buffer.make_dataset()
        
        if verbose:
            log.info(f"obs shape: {obs.shape}")                 # obs shape: torch.Size([20000, 1, 11])
            log.info(f"chains shape: {chains.shape}")           # torch.Size([20000, 11, 4, 3])
            log.info(f"returns shape: {returns.shape}")         # torch.Size([20000])
            log.info(f"oldvalues shape: {oldvalues.shape}")     # torch.Size([20000])
            log.info(f"advantages shape: {advantages.shape}")   # torch.Size([20000])
            log.info(f"oldlogprobs shape: {oldlogprobs.shape}") # torch.Size([20000, 10, 4, 3])
        
        
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
                    {k: obs[k][batch_inds_b] for k in obs},            # obs_b            visual inputs and states input. overload 
                    chains[batch_inds_b, denoising_inds_b],            # chains_prev_b    self.batch_size x self.horizon_steps x self.act_dim == [B, 4, 3] for hopper
                    chains[batch_inds_b, denoising_inds_b + 1],        # chains_next_b    self.batch_size x self.horizon_steps x self.act_dim == [B, 4, 3] B=50_000
                    denoising_inds_b,                                  # denoising_inds_b [B,] = [50_000,] 
                    returns[batch_inds_b],                             # returns_b        [B,] = [50_000,] 
                    oldvalues[batch_inds_b],                           # values_b         [B,] = [50_000,] 
                    advantages[batch_inds_b],                          # advantages_b     [B,] = [50_000,]  there are many duplicated entries.
                    oldlogprobs[batch_inds_b, denoising_inds_b]        # logprobs_b       self.batch_size x self.ft_denoising_step x self.horizon_steps x self.act_dim == [50_000, 10, 4, 3]
                )
                if verbose:
                    log.info(f"returns[batch_inds_b]={returns[batch_inds_b].shape}, oldvalues[batch_inds_b]={oldvalues[batch_inds_b].shape}, \
                        advantages[batch_inds_b]={advantages[batch_inds_b].shape},\
                            oldlogprobs[batch_inds_b]={oldlogprobs[batch_inds_b].shape}")
                
                # minibatch gradient descent
                self.model: PPODiffusion
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, eta = self.model.loss(*minibatch, 
                                                                                                            use_bc_loss=self.use_bc_loss, 
                                                                                                            reward_horizon=self.reward_horizon
                                                                                                            )
                if update_epoch ==0  and batch_id ==0 and np.abs(ratio-1.00)> 1e-6:
                    print(f"WARNING: ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
                
                clipfracs_list += [clipfrac]
                
                
                # bug fix: gradient accumulation
                loss.backward()
                if (batch_id + 1) % self.grad_accumulate == 0:
                    # update nets
                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor_ft.parameters(),
                                self.max_grad_norm,
                            )
                        self.actor_optimizer.step()
                        if (
                            self.learn_eta
                            and batch_id % self.eta_update_interval == 0
                        ):
                            self.eta_optimizer.step()
                    self.critic_optimizer.step()
                    # zero grad
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    if self.learn_eta:
                        self.eta_optimizer.zero_grad()
                    # report
                    log.info(f"run grad update at batch {batch_id}")
                    log.info(
                        f"approx_kl: {approx_kl}, update_epoch: {update_epoch}/{self.update_epochs}, num_batch: {self.total_steps //self.batch_size}"
                    )
                
                if verbose:
                    log.info(f"update_epoch: {update_epoch}, num_batch: {max(1, self.total_steps // self.batch_size)}, approx_kl: {approx_kl}")
                
                if (
                    self.target_kl is not None 
                    and approx_kl > self.target_kl 
                    and self.itr >= self.n_critic_warmup_itr  # bug fix
                ):
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
    
    # bugfix: overload, obtain logprobs from augmented observations in the buffer, instead of right after sampling.
    def get_samples(self, cond:dict, device='cpu'):
        samples = self.model.forward(cond=cond, deterministic=self.eval_mode, return_chain=True)
        action_samples = samples.trajectories                               # n_envs , horizon_steps , act_dim
        chains_venv = samples.chains                                        # n_envs , ft_denoising_steps+1 , horizon_steps , act_dim = (40, 11, 4, 3)
        if device=='cpu':
            return action_samples.cpu().numpy(), chains_venv.cpu().numpy()
        else:
            return action_samples.cpu().numpy(), chains_venv  # action_samples are still numpy because mujoco engine receives np.
    
    def run(self):
        self.init_buffer()
        self.prepare_run()
        self.buffer.reset() # needs overriding
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.reset_env(buffer_device=self.buffer_device)
            self.buffer.update_full_obs()### buffer needs to add new obs buffer.
            for step in tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps):
                if not self.verbose and step %100 ==0:
                    print(f"Processed step {step} /{self.n_steps}")
                with torch.no_grad():
                    ####### raw visual input. different from state input. ################
                    cond = {
                        key: torch.from_numpy(self.prev_obs_venv[key])
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }
                    # overload
                    action_samples, chains_venv= self.get_samples(cond=cond, device=self.buffer_device)
                # Apply multi-step action
                action_venv = action_samples[:, : self.act_steps]
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = \
                    self.venv.step(action_venv)
                
                # overload
                self.buffer.add(step, self.prev_obs_venv, chains_venv, reward_venv, terminated_venv, truncated_venv)
                
                self.prev_obs_venv = obs_venv
                self.cnt_train_step+= self.n_envs * self.act_steps if not self.eval_mode else 0
            
            self.buffer.summarize_episode_reward()
            if not self.eval_mode:
                ### bug fix
                self.buffer: PPODiffusionImgBufferGPU
                self.buffer.update_img(obs_venv, self.model)
                self.agent_update()
            
            self.adjust_finetune_schedule()
            self.log(train_prt_str_additional=f"diffusion - min sampling std:{self.diffusion_min_sampling_std}")
            
            self.update_lr()
            if self.itr >= self.n_critic_warmup_itr and self.learn_eta:
                self.eta_lr_scheduler.step()
            # update finetune scheduler of Diffusion Policy
            self.save_model()
            
            self.itr += 1
            
            
            
            