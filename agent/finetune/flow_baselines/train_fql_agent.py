# The MIT License (MIT)

# Copyright (c) 2025 FQL Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import pickle
import numpy as np
import torch
import logging
import wandb
import hydra
from collections import deque
from typing import Tuple
from torch import Tensor
log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_agent import TrainAgent
from model.flow.ft_baselines.fql import FQLModel
from agent.dataset.sequence import StitchedSequenceQLearningDataset
from itertools import chain
from tqdm import tqdm as tqdm

class TrainFQLAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Load offline dataset
        self.dataset_offline = hydra.utils.instantiate(cfg.offline_dataset)

        # Discount factor applied every act_steps
        self.gamma = cfg.train.gamma

        # Optimizers for actor and critic
        self.only_optimize_bc_flow=cfg.only_optimize_bc_flow
        log.info(f"self.only_optimize_bc_flow={self.only_optimize_bc_flow}.")
        
        if self.only_optimize_bc_flow:
            self.bc_actor_optimizer = torch.optim.Adam(
                self.model.bc_flow.parameters(),
                lr=cfg.train.actor_lr,
            )
            self.onestep_actor_optimizer = torch.optim.Adam(
                self.model.actor.parameters(),
                lr=cfg.train.actor_lr,
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
            chain(self.model.bc_flow.parameters(), self.model.actor.parameters()),
            lr=cfg.train.actor_lr,
        )
            
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
        )

        # Target network update rate
        self.target_ema_rate = cfg.train.target_ema_rate

        # Reward scaling factor
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Update frequencies. for fql they use 1:1 ratio between actor and critic
        self.critic_update_freq = int(cfg.train.batch_size / cfg.train.critic_replay_ratio)  # default is 1
        self.actor_update_freq = int(cfg.train.batch_size / cfg.train.actor_replay_ratio)    # default is 1
        self.actor_update_number= cfg.train.actor_update_repeat                              # default is 1
        
        # Buffer size for online data
        self.buffer_size = cfg.train.buffer_size

        # offline training steps: 
        self.offline_steps = cfg.train.offline_steps
        self.online_steps = cfg.train.online_steps
        if self.n_train_itr !=self.offline_steps+self.online_steps:
            raise ValueError(f"self.n_train_itr={self.n_train_itr}!=self.offline_steps({self.offline_steps})+self.online_steps({self.online_steps})")

        # Number of evaluation episodes
        self.n_steps_eval = cfg.train.n_steps_eval
        self.eval_base_model=cfg.train.eval_base_model
        
        # distillation loss weight (need to be tuned for each environment) 
        self.alpha= cfg.train.alpha
        
        # Model and device setup
        self.model: FQLModel
        self.device = cfg.get('device', 'cuda:7')
        self.model.to(self.device)

    def agent_update(self, batch:Tuple[dict, Tensor, dict, Tensor, Tensor]):
        cond_b, actions_b, next_cond_b, reward_b, terminated_b=batch
        
        loss_actor = 0.0

        # Update critic
        loss_critic, loss_critic_info = self.model.loss_critic(
            *batch,
            self.gamma,
        )
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        # Update target critic
        self.model.update_target_critic(self.target_ema_rate)

        # Update actor if frequency condition met. for fql they use 1:1 update ratio between actor and critic.
        if self.only_optimize_bc_flow:
            loss_bc_flow = self.model.loss_bc_flow(cond_b, actions_b)
            self.bc_actor_optimizer.zero_grad()
            loss_bc_flow.backward()
            self.bc_actor_optimizer.step()
            # place holder. this time loss_actor only records bc_flow loss.
            loss_actor = 0.00
            loss_actor_info={
            'loss_actor': 0.00,
            'loss_bc_flow': loss_bc_flow.item(),
            'q_loss': 0.00,
            'distill_loss': 0.00,
            'q': 0.00,
            'onestep_expert_bc_loss': 0.0
        }
        else:
            loss_actor, loss_actor_info = self.model.loss_actor(cond_b, actions_b, self.alpha)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
        
        return loss_critic, loss_actor, loss_critic_info, loss_actor_info
        
    def run(self):
        # Initialize online FIFO replay buffers
        self.obs_buffer = deque(maxlen=self.buffer_size)
        self.next_obs_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffer = deque(maxlen=self.buffer_size)
        self.terminated_buffer = deque(maxlen=self.buffer_size)
        loss_critic_info={}
        loss_actor_info={}
        
        # Load offline dataset into numpy arrays for efficient sampling
        dataloader_offline: StitchedSequenceQLearningDataset = torch.utils.data.DataLoader(
            self.dataset_offline,
            batch_size=len(self.dataset_offline),
            drop_last=False,
        )
        # Get dataset size and shapes
        dataset_size = len(self.dataset_offline)
        log.info(f"dataset_size={dataset_size}")
        obs_dim = self.dataset_offline[0][1]["state"].shape[-1]  # Shape of state
        cond_steps = self.dataset_offline[0][1]["state"].shape[-2]  # Horizon of state
        act_steps = self.dataset_offline[0][0].shape[0]  # Number of action steps
        act_dim = self.dataset_offline[0][0].shape[-1]  # Action dimension
        assert act_dim == self.cfg.action_dim
        assert act_steps == self.cfg.act_steps
        assert obs_dim == self.cfg.obs_dim
        assert cond_steps == self.cfg.cond_steps
        
        log.info(f"Caching dataset into numpy arrays with dataset_size={dataset_size}, obs_dim={obs_dim}, act_steps={act_steps}, act_dim={act_dim}")
        # Pre-allocate NumPy arrays
        obs_buffer_off = np.empty((dataset_size, cond_steps, obs_dim), dtype=np.float32)
        next_obs_buffer_off = np.empty((dataset_size, cond_steps, obs_dim), dtype=np.float32)
        action_buffer_off = np.empty((dataset_size, act_steps, act_dim), dtype=np.float32)
        reward_buffer_off = np.empty(dataset_size, dtype=np.float32)
        terminated_buffer_off = np.empty(dataset_size, dtype=np.float32)
        assert self.batch_size < len(obs_buffer_off)
        
        # Copy batches into pre-allocated arrays
        start_idx = 0
        for batch in dataloader_offline:
            actions, states_and_next, rewards, terminated = batch
            states = states_and_next["state"]  # Shape: (batch, obs_dim)
            next_states = states_and_next["next_state"]  # Shape: (batch, obs_dim)
            batch_size = states.shape[0]
            end_idx = start_idx + batch_size
            # Copy data directly into arrays
            obs_buffer_off[start_idx:end_idx] = states.cpu().numpy()# Shape: (N_off, obs_dim)
            next_obs_buffer_off[start_idx:end_idx] = next_states.cpu().numpy()# Shape: (N_off, obs_dim)
            action_buffer_off[start_idx:end_idx] = actions.cpu().numpy()# Shape: (N_off, act_dim)
            reward_buffer_off[start_idx:end_idx] = rewards.cpu().numpy().flatten() # Shape: (N_off,)
            terminated_buffer_off[start_idx:end_idx] = terminated.cpu().numpy().flatten() # Shape: (N_off,)
            start_idx = end_idx
        log.info(f"Finished caching dataset into numpy arrays with dataset_size={dataset_size}, obs_dim={obs_dim}, act_steps={act_steps}, act_dim={act_dim}. Sampling starts.")

        # Training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        self.success_rate=0.0
        self.avg_episode_reward=0.0
        self.avg_best_reward=0.0
        self.num_episode_finished=0.0
        self.avg_traj_length=0.0
        if self.eval_base_model:
            self.success_rate_base_model=0.0
            self.avg_episode_reward_base_model=0.0
            self.avg_best_reward_base_model=0.0
            self.num_episode_finished_base_model=0.0
            self.avg_traj_length_base_model=0.0
        
        while self.itr < self.n_train_itr:
            if self.itr % 1000 == 0:
                if self.itr  <= self.offline_steps:
                    print(f"Finished training iteration {self.itr} of {self.n_train_itr}. Offline training (total offline itrs={self.offline_steps})")
                else:
                    print(f"Finished training iteration {self.itr} of {self.n_train_itr}. Off2On training")
            # Prepare video paths for rendering
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )
            # Set train or eval mode
            eval_mode = (
                self.itr % self.val_freq == 0
                and not self.force_train
            ) or self.itr==0
            self.model.eval() if eval_mode else self.model.train()
            
            if eval_mode or self.itr == 0 or self.reset_at_iteration:
                self.prev_obs_venv = self.reset_env_all(options_venv=options_venv)
            
            if eval_mode:
                # Evaluation
                self.evaluate(self.prev_obs_venv, mode='onestep')
                if self.eval_base_model: 
                    self.evaluate(self.prev_obs_venv, mode='base_model')
            else:
                if self.itr < self.offline_steps:
                    # Offline RL.
                    n_offline = self.batch_size
                    inds_off = np.random.choice(len(obs_buffer_off), n_offline, replace=True)
                    obs_b_off = torch.from_numpy(obs_buffer_off[inds_off]).float().to(self.device)
                    actions_b_off = torch.from_numpy(action_buffer_off[inds_off]).float().to(self.device)
                    next_obs_b_off = torch.from_numpy(next_obs_buffer_off[inds_off]).float().to(self.device)
                    rewards_b_off = torch.from_numpy(reward_buffer_off[inds_off]).float().to(self.device)
                    terminated_b_off = torch.from_numpy(terminated_buffer_off[inds_off]).float().to(self.device)
                    batch=(
                        {"state": obs_b_off},
                        actions_b_off,
                        {"state": next_obs_b_off},
                        rewards_b_off,
                        terminated_b_off
                    )
                    # update agent with this batch.
                    loss_critic, loss_actor, loss_critic_info, loss_actor_info=self.agent_update(batch)
                else:
                    # Online rollout
                    for step in range(self.n_steps):
                        with torch.no_grad():
                            cond = {
                                "state": torch.from_numpy(self.prev_obs_venv["state"]).float().to(self.device)
                            }
                            samples = self.model.forward(cond=cond, mode='onestep').cpu().numpy()  # Shape: (n_env, horizon, act_dim)
                        action_venv = samples[:, :self.act_steps]  # Shape: (n_env, act_steps, act_dim)
                        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                        done_venv = terminated_venv | truncated_venv
                        # Store transitions in online buffer
                        for i in range(self.n_envs):
                            self.obs_buffer.append(self.prev_obs_venv["state"][i])  # Shape: (obs_dim,)
                            if "final_obs" in info_venv[i]:
                                self.next_obs_buffer.append(info_venv[i]["final_obs"]["state"])
                            else:
                                self.next_obs_buffer.append(obs_venv["state"][i])
                            self.action_buffer.append(action_venv[i])  # Shape: (act_steps, act_dim)
                        self.reward_buffer.extend((reward_venv * self.scale_reward_factor).tolist())
                        self.terminated_buffer.extend(terminated_venv.tolist())
                        
                        self.prev_obs_venv = obs_venv
                        cnt_train_step += self.n_envs * self.act_steps
                    # Sample half from offline and half from online data
                    n_offline = self.batch_size // 2
                    n_online = self.batch_size - n_offline
                    if n_online < len(self.obs_buffer):
                        # Offline sampling
                        inds_off = np.random.choice(len(obs_buffer_off), n_offline, replace=True)
                        obs_b_off = torch.from_numpy(obs_buffer_off[inds_off]).float().to(self.device)
                        next_obs_b_off = torch.from_numpy(next_obs_buffer_off[inds_off]).float().to(self.device)
                        actions_b_off = torch.from_numpy(action_buffer_off[inds_off]).float().to(self.device)
                        rewards_b_off = torch.from_numpy(reward_buffer_off[inds_off]).float().to(self.device)
                        terminated_b_off = torch.from_numpy(terminated_buffer_off[inds_off]).float().to(self.device)
                        # Online sampling
                        inds_on = np.random.choice(len(self.obs_buffer), n_online, replace=False)
                        obs_b_on = torch.from_numpy(np.array([self.obs_buffer[i] for i in inds_on])).float().to(self.device)
                        next_obs_b_on = torch.from_numpy(np.array([self.next_obs_buffer[i] for i in inds_on])).float().to(self.device)
                        actions_b_on = torch.from_numpy(np.array([self.action_buffer[i] for i in inds_on])).float().to(self.device)
                        rewards_b_on = torch.from_numpy(np.array([self.reward_buffer[i] for i in inds_on])).float().to(self.device)
                        terminated_b_on = torch.from_numpy(np.array([self.terminated_buffer[i] for i in inds_on])).float().to(self.device)
                        # Combine samples
                        obs_b = torch.cat([obs_b_off, obs_b_on], dim=0)  # Shape: (batch_size, obs_dim)
                        next_obs_b = torch.cat([next_obs_b_off, next_obs_b_on], dim=0)
                        actions_b = torch.cat([actions_b_off, actions_b_on], dim=0)  # Shape: (batch_size, act_steps, act_dim)
                        rewards_b = torch.cat([rewards_b_off, rewards_b_on], dim=0)  # Shape: (batch_size,)
                        terminated_b = torch.cat([terminated_b_off, terminated_b_on], dim=0)
                        batch=(
                            {"state": obs_b},
                            actions_b,
                            {"state": next_obs_b},
                            rewards_b,
                            terminated_b,
                        )
                        # update agent with this batch.
                        loss_critic, loss_actor, loss_critic_info, loss_actor_info=self.agent_update(batch)
            
            # Save model periodically
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log metrics
            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                time = timer()
                if eval_mode:
                    log.info(
                        f"eval (one step model): success rate {self.success_rate:8.4f} | avg episode reward {self.avg_episode_reward:8.4f} | avg best reward {self.avg_best_reward:8.4f}"
                    )
                    if self.eval_base_model:
                        log.info(
                        f"eval (base model): success rate {self.success_rate_base_model:8.4f} | avg episode reward {self.avg_episode_reward_base_model:8.4f} | avg best reward {self.avg_best_reward_base_model:8.4f}"
                    )
                    eval_log_dict={
                            "success rate - eval": self.success_rate,
                            "avg episode reward - eval": self.avg_episode_reward,
                            "avg best reward - eval": self.avg_best_reward,
                            "num episode - eval": self.num_episode_finished,
                            "avg traj length - eval": self.avg_traj_length,
                        }
                    if self.eval_base_model:
                        eval_log_dict.update({
                            "success rate (base model) - eval": self.success_rate_base_model,
                            "avg episode reward(base model) - eval": self.avg_episode_reward_base_model,
                            "avg best reward (base model)- eval": self.avg_best_reward_base_model,
                            "num episode (base model)- eval": self.num_episode_finished_base_model,
                            "avg traj length (base model) - eval": self.avg_traj_length_base_model,
                        })
                    run_results[-1].update(eval_log_dict)
                    if self.use_wandb:
                        wandb.log(
                            eval_log_dict,
                            step=self.itr,
                            commit=False,
                        )
                else:
                    # for SAC like algorithms witih only 1 environment and collects reward for only 1 step during training, this one-step reward is somewhat meaningless. So we just don't print it. 
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | loss critic {loss_critic:8.4f} | t {time:8.4f}"
                    )
                    train_log_dict = {
                        "total env step": cnt_train_step,
                        "loss - critic": loss_critic,
                        "num episode - train": self.num_episode_finished,
                    }
                    if loss_actor:
                        train_log_dict["loss - actor"] = loss_actor
                    train_log_dict.update(loss_critic_info)
                    train_log_dict.update(loss_actor_info)
                    if self.use_wandb:
                        wandb.log(train_log_dict, step=self.itr, commit=True)
                    run_results[-1].update(train_log_dict)
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
    
    def evaluate(self, prev_obs_venv:dict, mode:str='onestep'):  
        """evaluate onestep policy or the multistep base policy. 
        """
        assert mode in ['onestep', 'base_model']
        print(f"Evaluating...")
        # Reset environments during evaluation
        firsts_trajs = np.zeros((self.n_steps_eval + 1, self.n_envs))
        firsts_trajs[0] = 1
        # Online rollout starts
        reward_trajs = np.zeros((self.n_steps_eval, self.n_envs))
        for step in range(self.n_steps_eval):
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                }
                samples = self.model.forward(cond=cond,mode=mode).cpu().numpy()  # Shape: (n_env, horizon, act_dim)
            action_venv = samples[:, :self.act_steps]  # Shape: (n_env, act_steps, act_dim)
            # Step environment
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            done_venv = terminated_venv | truncated_venv
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            prev_obs_venv = obs_venv
        # Compute episode rewards
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start, end = env_steps[i], env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if episodes_start_end:
            reward_trajs_split = [reward_trajs[start:end + 1, env_ind] for env_ind, start, end in episodes_start_end]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array([np.sum(traj) for traj in reward_trajs_split])
            episode_best_reward = np.array([np.max(traj) / self.act_steps for traj in reward_trajs_split])
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
            avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        else:
            num_episode_finished = 0
            avg_episode_reward = avg_best_reward = success_rate = 0
            avg_traj_length = 0
        
        # record
        if mode == 'onestep':
            self.avg_episode_reward=avg_episode_reward
            self.avg_best_reward=avg_best_reward
            self.success_rate=success_rate
            self.num_episode_finished=num_episode_finished
            self.avg_traj_length=avg_traj_length
        elif mode=='base_model':
            self.avg_episode_reward_base_model=avg_episode_reward
            self.avg_best_reward_base_model=avg_best_reward
            self.success_rate_base_model=success_rate
            self.num_episode_finished_base_model=num_episode_finished
            self.avg_traj_length_base_model=avg_traj_length
        else:
            raise ValueError(f"unsupported mode={mode}. A valid choice must be in ['onstep', 'base_model']")