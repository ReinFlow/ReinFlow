import numpy as np
import torch
import logging
log = logging.getLogger(__name__)
from util.reward_scaling import RunningRewardScaler
from util.reward_scaling_ts import RunningRewardScalerTensor
from collections import deque
from model.common.critic import ViTCritic
from model.diffusion.diffusion_ppo import PPODiffusion
from model.flow.ft_ppo.ppoflow import PPOFlow

class PPOBuffer:
    '''
    on-policy buffer for state inputs and cpu storage.
    '''
    def __init__(self,
                 n_steps,
                 n_envs, 
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device):
        
        self.n_steps = n_steps
        self.n_envs  = n_envs
        
        self.horizon_steps = horizon_steps
        self.act_steps = act_steps
        self.action_dim = action_dim
        
        self.n_cond_step = n_cond_step 
        self.obs_dim  =obs_dim
         
        self.furniture_sparse_reward =furniture_sparse_reward  
        self.best_reward_threshold_for_success=best_reward_threshold_for_success
        self.save_full_observation= save_full_observation
        self.reward_scale_running =reward_scale_running
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(num_envs=n_envs, gamma=gamma)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reward_scale_const = reward_scale_const
        self.device = device
    
    def reset(self):
        self.obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
        self.samples_trajs = np.zeros((self.n_steps, self.n_envs, self.horizon_steps, self.action_dim))
        
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs))
    
    def add(self, step, state_venv, output_actions_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv):
        done_venv = terminated_venv | truncated_venv
        
        self.obs_trajs["state"][step] = state_venv
        self.samples_trajs[step] = output_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv
        
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv
    
    
    @torch.no_grad
    def update_adv_returns(self, obs_venv, critic:torch.nn.Module, buffer_device='cpu'): 
        '''
        obs_venv: dict containing numpy.ndarray
        '''
        # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
        obs_venv_ts = {
            "state": torch.from_numpy(obs_venv["state"])
            .float()
            .to(self.device)
        }
        
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs)) if buffer_device == 'cpu' else torch.zeros(self.n_steps, self.n_envs, device=self.device)
        
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                nextvalues = critic.forward(obs_venv_ts).reshape(1, -1)
                nextvalues = nextvalues.cpu().numpy() if buffer_device == 'cpu' else nextvalues.to(self.device)
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        self.returns_trajs = self.advantages_trajs + self.value_trajs
    
    def make_dataset(self):
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1)
        samples= torch.tensor(self.samples_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)
        return (obs, samples, returns, values, advantages, logprobs)
    
    @torch.no_grad
    def update(self, obs_venv:dict, critic:torch.nn.Module, device='cpu'):
        '''
        obs_venv: dict containing numpy.ndarray
        '''
        # normalize reward with running variance
        self.normalize_reward()
        self.update_adv_returns(obs_venv, critic, device)

    def update_full_obs(self):
        if self.save_full_observation:
            self.obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            self.obs_full_trajs = np.vstack((self.obs_full_trajs, self.prev_obs_venv["state"][:, -1][None]))
    
    def save_full_obs(self, info_venv):
        if self.save_full_observation:
            obs_full_venv = np.array([info["full_obs"]["state"] for info in info_venv])  # n_envs x act_steps x obs_dim
            self.obs_full_trajs = np.vstack((self.obs_full_trajs, obs_full_venv.transpose(1, 0, 2)))
    
    @torch.no_grad
    def normalize_reward(self):
        '''
        normalize self.reward_trajs
        '''
        if self.reward_scale_running:
            reward_trajs_transpose = self.running_reward_scaler(
                reward=self.reward_trajs.T, first=self.firsts_trajs[:-1].T
            )
            self.reward_trajs = reward_trajs_transpose.T
    
    @torch.no_grad
    def get_explained_var(self, values, returns):
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)
        return explained_var
   
    
    @torch.no_grad()
    def summarize_episode_reward(self):
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(self.firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                self.reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            
            # Calculate standard deviations
            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            
            # Calculate average length of valid episodes and its standard deviation
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end])*self.act_steps # account for multiple steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
            
        else:
            episode_reward = np.array([])
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.avg_best_reward = 0
            self.success_rate = 0
            self.avg_episode_length = 0.0
            self.std_episode_reward = 0
            self.std_best_reward = 0
            self.std_success_rate = 0
            self.std_episode_length = 0.0
            log.info("[WARNING] No episode completed within the iteration!")
        
class PPODiffusionBuffer(PPOBuffer):
    def __init__(self, 
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device
                 ):
        super().__init__(
                 n_steps,
                 n_envs, 
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
        self.ft_denoising_steps = n_ft_denoising_steps
    # overload
    def reset(self):
        self.obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
        self.chains_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        
        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim))   
    
    # overload
    def add(self, step, state_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv):
        self.obs_trajs["state"][step] = state_venv
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv = terminated_venv | truncated_venv
        
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv
    # overload
    def make_dataset(self):
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1)
        chains= torch.tensor(self.chains_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)
        return obs, chains, returns, values, advantages, logprobs

class PPODiffusionBufferGPU(PPODiffusionBuffer):
    def __init__(self, 
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
        # overload 
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScalerTensor(num_envs=n_envs, gamma=gamma, device = self.device)
        self.ft_denoising_steps = n_ft_denoising_steps
    
    # overload
    def reset(self):
        # when created on cpu you can also pin_memory=True. but it does not work when created on gpu.
        self.obs_trajs = {
            "state": torch.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim), dtype=torch.float32, device=self.device)
        }
        self.chains_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device)

        self.value_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
    
    # overload
    def make_dataset(self):
        obs = self.obs_trajs["state"].flatten(0,1)
        chains= self.chains_trajs.flatten(0,1)
        returns = self.returns_trajs.flatten(0,1)
        values = self.value_trajs.flatten(0,1)
        advantages = self.advantages_trajs.flatten(0,1)
        logprobs = self.logprobs_trajs.flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs
    
    # overload
    def add(self, step, state_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv):
        self.obs_trajs["state"][step] = torch.from_numpy(state_venv).float().to(self.device)
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = torch.from_numpy(reward_venv).float().to(self.device)
        self.terminated_trajs[step] = torch.from_numpy(terminated_venv).float().to(self.device)
        self.firsts_trajs[step + 1] = torch.from_numpy(terminated_venv | truncated_venv).float().to(self.device) # done_venv
        
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv
    
    # overload
    @torch.no_grad
    def get_explained_var(self, values, returns):
        # Assuming values and returns are already tensors
        y_pred = values.detach()  # Detach to prevent gradient tracking
        y_true = returns.detach()
        var_y = y_true.var().item()
        explained_var = (float('nan') if var_y == 0 else 1 - ((y_true - y_pred).var().item() / var_y))
        return explained_var  # Returns a floating point number
    
    # overload, for GPU version
    @torch.no_grad
    def summarize_episode_reward(self):
        episodes_start_end = []
        # Convert firsts_trajs to numpy for processing
        firsts_trajs_np = self.firsts_trajs.cpu().numpy()  

        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        
        if len(episodes_start_end) > 0:
            # Select reward_trajs using numpy slicing
            reward_trajs_split = [
                self.reward_trajs[start:end + 1, env_ind].cpu().numpy()
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            
            # Calculating episode_reward using numpy
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
            
            if self.furniture_sparse_reward:  # For furniture tasks
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            # Compute metrics
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            # Calculate standard deviations
            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            # Calculate average length of valid episodes and its standard deviation
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end])*self.act_steps # account for multiple steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
            
        else:
            episode_reward = np.array([])
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.std_episode_reward = 0
            self.avg_best_reward = 0
            self.std_best_reward = 0
            self.success_rate = 0
            self.std_success_rate = 0
            self.avg_episode_length = 0
            self.std_episode_length = 0
            log.info("[WARNING] No episode completed within the iteration!")

class PPODiffusionImgBuffer(PPODiffusionBuffer):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 aug, # image augmentation
                 fix_nextvalue_augment_bug,
                 device):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device
        )
        
        self.aug=aug 
        self.fix_nextvalue_augment_bug=fix_nextvalue_augment_bug
    # overload
    def reset(self):
        # visual inputs
        self.obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k])  # (s, e, t, C, H, W)
                )
                for k in self.obs_dim
            }
        self.chains_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        
        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim))   
    
    # overload. compute value and logprob only during updates, where they are computed on augmented samples. 
    def add(self, step, prev_obs_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv):
        # visual inputs: rgb, state
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = prev_obs_venv[k]
        
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = terminated_venv | truncated_venv # done_venv
    
    # bugfix: overload 
    @torch.no_grad
    def update_img(self, obs_venv:dict, model:PPODiffusion):
        '''
        bugfix:
        '''
        # normalize reward with running variance
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_adv_returns(obs_venv, model.critic)
    
    # bugfix: overload 
    @torch.no_grad
    def update_value_logprob(self, model:PPODiffusion):
        # bugfix: image augmentation on the whole observation trajectory
        '''bug fix'''
        
        obs_trajs_ts = {
            key: torch.from_numpy(self.obs_trajs[key]).float().to(self.device) 
            for key in self.obs_dim
        }
        if self.aug:
            rgb = obs_trajs_ts["rgb"].flatten(0,2) # (s x e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_trajs_ts["rgb"]=rgb.reshape(self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute values and (old)logprobs
        for step in range(self.n_steps):
            cond = {
                key: obs_trajs_ts[key][step].to(self.device) for key in self.obs_dim
            }
            chains_venv = torch.from_numpy(self.chains_trajs[step]).float().to(self.device)
            
            critic: ViTCritic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).cpu().numpy().flatten()
            
            self.logprobs_trajs[step] = model.get_logprobs(cond, chains_venv).reshape(self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim).cpu().numpy()

    # bugfix: overload
    @torch.no_grad
    def update_adv_returns(self, obs_venv, critic:ViTCritic): 
        '''
        Do image augmentation on the whole buffer, and compute value, logprobabilities, adv, and return on those augmentated data.
        obs_venv: dict containing numpy.ndarray
        '''
        # overload to accept rgbs in the last observation
        '''
        potential improvement: add image augmentation to this last obs and then use in GAE
        '''
        obs_venv_ts = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }
        if self.fix_nextvalue_augment_bug and self.aug:
            rgb = obs_venv_ts["rgb"].flatten(0,1) # (e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_venv_ts["rgb"]=rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute adv
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                '''
                bug fix.
                the critic is ViTCritic and the forward method, by default, adds an augmentation to inputs. However, since we already did the augmentation in the buffer, the inner augmentation in the critic should be turned off
                '''
                nextvalues = critic.forward(obs_venv_ts, no_augment=True).reshape(1, -1) # no augmentation added. 
                nextvalues = nextvalues.cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        # compute return
        self.returns_trajs = self.advantages_trajs + self.value_trajs
    
    # bugfix: overload
    @torch.no_grad
    def make_dataset(self):
        '''
        bug fix:
        '''
        obs = {
            "state": torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1),
            "rgb": torch.tensor(self.obs_trajs["rgb"], device=self.device).float().flatten(0,1)
        }
        chains= torch.tensor(self.chains_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs
    
class PPODiffusionImgBufferGPU(PPODiffusionBufferGPU):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 aug,
                 fix_nextvalue_augment_bug,
                 device
                 ):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
        
        self.aug=aug 
        self.fix_nextvalue_augment_bug=fix_nextvalue_augment_bug
    
    # overload
    def reset(self):
        # visual inputs
        self.obs_trajs = {
            k: torch.zeros((self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]), dtype=torch.float32, device=self.device)
            for k in self.obs_dim
        }# (s, e, t, C, H, W)
        self.chains_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device)

        self.value_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)

    # bugfix: overload
    def make_dataset(self):
        '''
        bug fix:
        '''
        obs = {
            "state": self.obs_trajs["state"].clone().detach().flatten(0,1),
            "rgb": self.obs_trajs["rgb"].clone().detach().flatten(0,1)
        }
        chains= self.chains_trajs.flatten(0,1)
        returns = self.returns_trajs.flatten(0,1)
        values = self.value_trajs.flatten(0,1)
        advantages = self.advantages_trajs.flatten(0,1)
        logprobs = self.logprobs_trajs.flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs
    
    # bugfix: overload
    def add(self, step, prev_obs_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv):
        # visual inputs: rgb, state
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = torch.from_numpy(prev_obs_venv[k]).float().to(self.device)
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = torch.from_numpy(reward_venv).float().to(self.device)
        self.terminated_trajs[step] = torch.from_numpy(terminated_venv).float().to(self.device)
        self.firsts_trajs[step + 1] = torch.from_numpy(terminated_venv | truncated_venv).float().to(self.device) # done_venv

    # bugfix: overload 
    @torch.no_grad
    def update_img(self, obs_venv:dict, model:PPODiffusion):
        '''
        bugfix:
        '''
        # normalize reward with running variance
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_adv_returns(obs_venv, model.critic)
    
    # bugfix: overload 
    @torch.no_grad
    def update_value_logprob(self, model):
        # bugfix image augmentation
        '''bug fix'''
        if self.aug:
            rgb = self.obs_trajs["rgb"].flatten(0,2) # (s x e x t, C, H, W)
            rgb = self.aug(rgb)
            self.obs_trajs["rgb"]=rgb.reshape(self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute values and (old)logprobs
        for step in range(self.n_steps):
            cond = {
                key: self.obs_trajs[key][step] for key in self.obs_dim
            }
            chains_venv = self.chains_trajs[step]
            
            critic:ViTCritic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).squeeze().float().to(self.device) # GPU version
            
            self.logprobs_trajs[step] = model.get_logprobs(cond, chains_venv).reshape(self.n_envs, self.ft_denoising_steps, self.horizon_steps, self.action_dim) # GPU version
        
    # bugfix: overload
    @torch.no_grad
    def update_adv_returns(self, obs_venv, critic:ViTCritic): 
        '''
        obs_venv: dict containing numpy.ndarray
        '''
        # this is different for image inputs.
        obs_venv_ts = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }
        if self.fix_nextvalue_augment_bug and self.aug:
            rgb = obs_venv_ts["rgb"].flatten(0,1) # (e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_venv_ts["rgb"]=rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute adv
        self.advantages_trajs = torch.zeros(self.n_steps, self.n_envs, device=self.device)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                '''
                bug fix.
                the critic is ViTCritic and the forward method, by default, adds an augmentation to inputs. However, since we already did the augmentation in the buffer, the inner augmentation in the critic should be turned off
                '''
                nextvalues = critic.forward(cond=obs_venv_ts, no_augment=True).reshape(1, -1)
                nextvalues = nextvalues.to(self.device)
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        # compute return
        self.returns_trajs = self.advantages_trajs + self.value_trajs


class PPOFlowBuffer(PPOBuffer):
    '''state input'''
    def __init__(self, 
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device):
        super().__init__(
                 n_steps,
                 n_envs, 
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
        self.ft_denoising_steps = n_ft_denoising_steps
    # overload
    def reset(self):
        self.obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
        self.chains_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        
        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs))
    
    # overload
    def add(self, step, state_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv):
        self.obs_trajs["state"][step] = state_venv
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = done_venv = terminated_venv | truncated_venv
        
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv
    
    # overload
    def make_dataset(self):
        obs = torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1)
        chains= torch.tensor(self.chains_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs

class PPOFlowBufferGPU(PPOFlowBuffer):
    def __init__(self, 
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device):
        super().__init__(
                 n_steps,    
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device
        )
        # overload 
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScalerTensor(num_envs=n_envs, gamma=gamma, device = self.device)
        self.ft_denoising_steps = n_ft_denoising_steps
        
        
        
        # overload
    def reset(self):
        # when created on cpu you can also pin_memory=True. but it does not work when created on gpu.
        self.obs_trajs = {
            "state": torch.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim), dtype=torch.float32, device=self.device)
        }
        self.chains_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device)

        self.value_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
    
    
    # overload
    def add(self, step, state_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv, value_venv, logprob_venv):
        self.obs_trajs["state"][step] = torch.from_numpy(state_venv).float().to(self.device)
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = torch.from_numpy(reward_venv).float().to(self.device)
        self.terminated_trajs[step] = torch.from_numpy(terminated_venv).float().to(self.device)
        self.firsts_trajs[step + 1] = torch.from_numpy(terminated_venv | truncated_venv).float().to(self.device) # done_venv
        
        self.value_trajs[step] = value_venv
        self.logprobs_trajs[step] = logprob_venv
        
    # overload
    def make_dataset(self):
        obs = self.obs_trajs["state"].flatten(0,1)
        chains= self.chains_trajs.flatten(0,1)
        returns = self.returns_trajs.flatten(0,1)
        values = self.value_trajs.flatten(0,1)
        advantages = self.advantages_trajs.flatten(0,1)
        logprobs = self.logprobs_trajs.flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs
    
    # overload
    @torch.no_grad
    def get_explained_var(self, values, returns):
        # Assuming values and returns are already tensors
        y_pred = values.detach()  # Detach to prevent gradient tracking
        y_true = returns.detach()
        var_y = y_true.var().item()
        explained_var = (float('nan') if var_y == 0 else 1 - ((y_true - y_pred).var().item() / var_y))
        return explained_var  # Returns a floating point number
    
    # overload, for GPU version
    @torch.no_grad
    def summarize_episode_reward(self):
        episodes_start_end = []
        # Convert firsts_trajs to numpy for processing
        firsts_trajs_np = self.firsts_trajs.cpu().numpy()  

        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs_np[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        
        if len(episodes_start_end) > 0:
            # Select reward_trajs using numpy slicing
            reward_trajs_split = [
                self.reward_trajs[start:end + 1, env_ind].cpu().numpy()
                for env_ind, start, end in episodes_start_end
            ]
            self.num_episode_finished = len(reward_trajs_split)
            
            # Calculating episode_reward using numpy
            episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split])
            
            if self.furniture_sparse_reward:  # For furniture tasks
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            # Compute metrics
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            # Calculate standard deviations
            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            # Calculate average length of valid episodes and its standard deviation
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end])*self.act_steps # account for multiple steps
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
            
        else:
            episode_reward = np.array([])
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.std_episode_reward = 0
            self.avg_best_reward = 0
            self.std_best_reward = 0
            self.success_rate = 0
            self.std_success_rate = 0
            self.avg_episode_length = 0
            self.std_episode_length = 0
            log.info("[WARNING] No episode completed within the iteration!")

# revised
class PPOFlowImgBuffer(PPOFlowBuffer):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 aug, # image augmentation
                 fix_nextvalue_augment_bug:bool,
                 device,
                 log_prob_cfg_dict:dict
                 ):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device
        )
        
        self.aug=aug 
        self.log_prob_cfg_dict=log_prob_cfg_dict
        self.fix_nextvalue_augment_bug=fix_nextvalue_augment_bug
    # overload
    def reset(self):
        # visual inputs
        self.obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k])  # (s, e, t, C, H, W)
                )
                for k in self.obs_dim
            }
        self.chains_trajs = np.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim))
        self.terminated_trajs = np.zeros((self.n_steps, self.n_envs))
        
        self.firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        self.reward_trajs = np.zeros((self.n_steps, self.n_envs))
        
        self.value_trajs = np.empty((self.n_steps, self.n_envs))
        self.logprobs_trajs = np.zeros((self.n_steps, self.n_envs)) # flow diffusion difference
    
    # overload. compute value and logprob only during updates, where they are computed on augmented samples. 
    def add(self, step, prev_obs_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv):
        # visual inputs: rgb, state
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = prev_obs_venv[k]
        
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = reward_venv
        self.terminated_trajs[step] = terminated_venv
        self.firsts_trajs[step + 1] = terminated_venv | truncated_venv # done_venv
    
    # bugfix: overload 
    @torch.no_grad
    def update_img(self, obs_venv:dict, model:PPOFlow):
        '''
        bugfix:
        '''
        # normalize reward with running variance
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_adv_returns(obs_venv, model.critic)
    
    # bugfix: overload 
    @torch.no_grad
    def update_value_logprob(self, model:PPOFlow):
        # bugfix: image augmentation on the whole observation trajectory
        '''bug fix'''
        
        obs_trajs_ts = {
            key: torch.from_numpy(self.obs_trajs[key]).float().to(self.device) 
            for key in self.obs_dim
        }
        if self.aug:
            rgb = obs_trajs_ts["rgb"].flatten(0,2) # (s x e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_trajs_ts["rgb"]=rgb.reshape(self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute values and (old)logprobs
        for step in range(self.n_steps):
            cond = {
                key: obs_trajs_ts[key][step].to(self.device) for key in self.obs_dim
            }
            chains_venv = torch.from_numpy(self.chains_trajs[step]).float().to(self.device)
            
            critic: ViTCritic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).cpu().numpy().flatten()
            
            self.logprobs_trajs[step] = model.get_logprobs(cond, chains_venv,
                                                           get_entropy=False,
                                                           get_chains_stds=False,
                                                           verbose_entropy_stats=True,
                                                           debug=True,
                                                           **self.log_prob_cfg_dict
                                                           ).cpu().numpy()

    # bugfix: overload
    @torch.no_grad
    def update_adv_returns(self, obs_venv, critic:ViTCritic): 
        '''
        Do image augmentation on the whole buffer, and compute value, logprobabilities, adv, and return on those augmentated data.
        obs_venv: dict containing numpy.ndarray
        '''
        # overload to accept rgbs in the last observation
        '''
        potential improvement: add image augmentation to this last obs and then use in GAE
        '''
        obs_venv_ts = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }
        # possible fix: also add augmentation to the last obs used in GAE
        if self.fix_nextvalue_augment_bug and self.aug:
            rgb = obs_venv_ts["rgb"].flatten(0,1) # (e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_venv_ts["rgb"]=rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
            
        # bugfix: compute adv
        self.advantages_trajs = np.zeros((self.n_steps, self.n_envs))
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                '''
                bug fix.
                the critic is ViTCritic and the forward method, by default, adds an augmentation to inputs. However, since we already did the augmentation in the buffer, the inner augmentation in the critic should be turned off
                '''
                nextvalues = critic.forward(obs_venv_ts, no_augment=True).reshape(1, -1) # no augmentation added. 
                nextvalues = nextvalues.cpu().numpy()
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        # compute return
        self.returns_trajs = self.advantages_trajs + self.value_trajs
    
    # bugfix: overload
    @torch.no_grad
    def make_dataset(self):
        '''
        bug fix:
        '''
        obs = {
            "state": torch.tensor(self.obs_trajs["state"], device=self.device).float().flatten(0,1),
            "rgb": torch.tensor(self.obs_trajs["rgb"], device=self.device).float().flatten(0,1)
        }
        chains= torch.tensor(self.chains_trajs, device=self.device).float().flatten(0,1)
        returns = torch.tensor(self.returns_trajs, device=self.device).float().flatten(0,1)
        values = torch.tensor(self.value_trajs, device=self.device).float().flatten(0,1)
        advantages = torch.tensor(self.advantages_trajs, device=self.device).float().flatten(0,1)
        logprobs = torch.tensor(self.logprobs_trajs, device=self.device).float().flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs

# revised
class PPOFlowImgBufferGPU(PPOFlowBufferGPU):
    def __init__(self,
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 aug,
                 fix_nextvalue_augment_bug:bool,
                 device,
                 log_prob_cfg_dict:dict
                 ):
        super().__init__(
                 n_steps,
                 n_envs, 
                 n_ft_denoising_steps,
                 horizon_steps, 
                 act_steps,
                 action_dim,
                 n_cond_step,
                 obs_dim,
                 save_full_observation,
                 furniture_sparse_reward,
                 best_reward_threshold_for_success,
                 reward_scale_running,
                 gamma,
                 gae_lambda,
                 reward_scale_const,
                 device)
        
        self.aug=aug 
        self.log_prob_cfg_dict=log_prob_cfg_dict
        self.fix_nextvalue_augment_bug=fix_nextvalue_augment_bug
    # overload
    def reset(self):
        # visual inputs
        self.obs_trajs = {
            k: torch.zeros((self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim[k]), dtype=torch.float32, device=self.device)
            for k in self.obs_dim
        }# (s, e, t, C, H, W)
        self.chains_trajs = torch.zeros((self.n_steps, self.n_envs, self.ft_denoising_steps + 1, self.horizon_steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.reward_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.terminated_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.firsts_trajs = torch.zeros((self.n_steps + 1, self.n_envs), dtype=torch.float32, device=self.device)

        self.value_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device)
        self.logprobs_trajs = torch.zeros((self.n_steps, self.n_envs), dtype=torch.float32, device=self.device) # flow -diffusion differnce

    # bugfix: overload
    def make_dataset(self):
        '''
        bug fix:
        '''
        obs = {
            "state": self.obs_trajs["state"].clone().detach().flatten(0,1),
            "rgb": self.obs_trajs["rgb"].clone().detach().flatten(0,1)
        }
        chains= self.chains_trajs.flatten(0,1)
        returns = self.returns_trajs.flatten(0,1)
        values = self.value_trajs.flatten(0,1)
        advantages = self.advantages_trajs.flatten(0,1)
        logprobs = self.logprobs_trajs.flatten(0,1)

        return obs, chains, returns, values, advantages, logprobs
    
    # bugfix: overload
    def add(self, step, prev_obs_venv, chains_actions_venv, reward_venv, terminated_venv, truncated_venv):
        # visual inputs: rgb, state
        for k in self.obs_trajs:
            self.obs_trajs[k][step] = torch.from_numpy(prev_obs_venv[k]).float().to(self.device)
        self.chains_trajs[step] = chains_actions_venv
        self.reward_trajs[step] = torch.from_numpy(reward_venv).float().to(self.device)
        self.terminated_trajs[step] = torch.from_numpy(terminated_venv).float().to(self.device)
        self.firsts_trajs[step + 1] = torch.from_numpy(terminated_venv | truncated_venv).float().to(self.device) # done_venv

    # bugfix: overload 
    @torch.no_grad
    def update_img(self, obs_venv:dict, model:PPOFlow):
        '''
        bugfix:
        '''
        # normalize reward with running variance
        self.normalize_reward()
        self.update_value_logprob(model)
        self.update_adv_returns(obs_venv, model.critic)
    
    # bugfix: overload 
    @torch.no_grad
    def update_value_logprob(self, model:PPOFlow):
        # bugfix image augmentation
        '''bug fix'''
        if self.aug:
            rgb = self.obs_trajs["rgb"].flatten(0,2) # (s x e x t, C, H, W)
            rgb = self.aug(rgb)
            self.obs_trajs["rgb"]=rgb.reshape(self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute values and (old)logprobs
        for step in range(self.n_steps):
            cond = {
                key: self.obs_trajs[key][step] for key in self.obs_dim
            }
            chains_venv = self.chains_trajs[step]
            
            critic:ViTCritic = model.critic
            self.value_trajs[step] = critic.forward(cond, no_augment=True).squeeze().float().to(self.device) # GPU version
            
            self.logprobs_trajs[step] = model.get_logprobs(cond, chains_venv,
                                                           get_entropy=False,
                                                           get_chains_stds=False,
                                                           verbose_entropy_stats=True,
                                                           debug=True,
                                                           **self.log_prob_cfg_dict
                                                           )# GPU version
    # bugfix: overload
    @torch.no_grad
    def update_adv_returns(self, obs_venv, critic:ViTCritic):
        '''
        obs_venv: dict containing numpy.ndarray
        '''
        # this is different for image inputs.
        obs_venv_ts = {
            key: torch.from_numpy(obs_venv[key]).float().to(self.device)
            for key in self.obs_dim
        }
        # possible fix: also add augmentation to the last obs used in GAE
        if self.fix_nextvalue_augment_bug and self.aug:
            rgb = obs_venv_ts["rgb"].flatten(0,1) # (e x t, C, H, W)
            rgb = self.aug(rgb)
            obs_venv_ts["rgb"]=rgb.reshape(self.n_envs, self.n_cond_step, *self.obs_dim['rgb'])
        
        # bugfix: compute adv
        self.advantages_trajs = torch.zeros(self.n_steps, self.n_envs, device=self.device)
        lastgaelam = 0
        for t in reversed(range(self.n_steps)):
            # get V(s_t+1)
            if t == self.n_steps - 1:
                '''
                bug fix.
                the critic is ViTCritic and the forward method, by default, adds an augmentation to inputs. However, since we already did the augmentation in the buffer, the inner augmentation in the critic should be turned off
                '''
                nextvalues = critic.forward(cond=obs_venv_ts, no_augment=True).reshape(1, -1)
                nextvalues = nextvalues.to(self.device)
            else:
                nextvalues = self.value_trajs[t + 1]
            # delta = r + gamma*V(st+1) - V(st)
            non_terminal = 1.0 - self.terminated_trajs[t]
            delta = (
                self.reward_trajs[t] * self.reward_scale_const
                + self.gamma * nextvalues * non_terminal
                - self.value_trajs[t]
            )
            # A = delta_t + gamma*lamdba*delta_{t+1} + ...
            self.advantages_trajs[t] = lastgaelam = (
                delta
                + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            )
        # compute return
        self.returns_trajs = self.advantages_trajs + self.value_trajs

