"""
Parent PPO fine-tuning agent class.
"""
from typing import Optional
import torch
import logging
from util.scheduler import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from util.scheduler_simple import CustomScheduler
log = logging.getLogger(__name__)
from agent.finetune.reinflow.train_agent import TrainAgent
from util.reward_scaling import RunningRewardScaler
from util.logging_custom import create_bordered_text
########################################################################
# appended by ReinFlow Authors
import os
import numpy as np
from util.timer import Timer
import os
import numpy as np
import pickle
import wandb
from util.reproducibility import set_seed_everywhere
########################################################################
class TrainPPOAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.resume = cfg.get('resume_path', False)
        
        # Batch size for logprobs calculations after an iteration --- 
        # prevent out of memory if using a single batch
        self.logprob_batch_size = cfg.train.get("logprob_batch_size", 10000)
        assert (
            self.logprob_batch_size % self.n_envs == 0
        ), "logprob_batch_size must be divisible by n_envs"

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Warm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_type = cfg.train.actor_lr_scheduler.get("type", "cosine")
        if self.actor_lr_type == "cosine":
            self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.actor_lr_type == "plateau":
            self.actor_lr_scheduler = WarmupReduceLROnPlateau(
                self.actor_optimizer,
                warmup_steps = cfg.train.actor_lr_scheduler.warmup_steps,
                target_lr = cfg.train.actor_lr,
                mode='max',         # Use 'max' for rewards if they are your metric
                min_lr = cfg.train.actor_lr_scheduler.min_lr,
                factor=0.6, 
                patience=4, 
                threshold=20,
                verbose=True
            )
        elif self.actor_lr_type=='constant_warmup':
            self.actor_lr_scheduler=CustomScheduler(self.actor_optimizer,
                                                    'constant_warmup', 
                                                  min=cfg.train.actor_lr_scheduler.min_lr, 
                                                  warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                                                  max=cfg.train.actor_lr)
        elif self.actor_lr_type=='cosine_custom':
            self.actor_lr_scheduler=CustomScheduler(self.actor_optimizer,
                                                     schedule_type='cosine', 
                                                     max=cfg.train.actor_lr,
                                                     hold_steps=cfg.train.actor_lr_scheduler.hold_steps,
                                                     anneal_steps=cfg.train.actor_lr_scheduler.anneal_steps,
                                                     min=cfg.train.actor_lr_scheduler.min_lr)
        else:
            raise ValueError(f"Invalid actor_lr_type: {self.actor_lr_type}")
        
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_type = cfg.train.critic_lr_scheduler.get("type", "cosine")
        if self.critic_lr_type == "cosine":
            self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.critic_optimizer,
                first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.critic_lr,
                min_lr=cfg.train.critic_lr_scheduler.min_lr,
                warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.critic_lr_type == "plateau":
            self.critic_lr_scheduler = WarmupReduceLROnPlateau(
                self.critic_optimizer,
                warmup_steps = cfg.train.critic_lr_scheduler.warmup_steps,
                target_lr = cfg.train.critic_lr,
                mode='max',         # Use 'max' for rewards if they are your metric
                min_lr = cfg.train.critic_lr_scheduler.min_lr,
                factor=0.6, 
                patience=4, 
                threshold=20,
                verbose=True
            )
        elif self.critic_lr_type=='constant_warmup':
            self.critic_lr_scheduler=CustomScheduler(self.critic_optimizer,
                                                     'constant_warmup', 
                                                        min=cfg.train.critic_lr_scheduler.min_lr, 
                                                        warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
                                                        max=cfg.train.critic_lr)
        elif self.critic_lr_type=='cosine_custom':
            self.critic_lr_scheduler=CustomScheduler(self.critic_optimizer,
                                                     schedule_type='cosine', 
                                                     max=cfg.train.critic_lr,
                                                     hold_steps=cfg.train.critic_lr_scheduler.hold_steps,
                                                     anneal_steps=cfg.train.critic_lr_scheduler.anneal_steps,
                                                     min=cfg.train.critic_lr_scheduler.min_lr)
        else:
            raise ValueError(f"Invalid critic_lr_type: {self.critic_lr_type}")
        
        self.visualize_lr(cfg)
        
        # Generalized advantage estimation
        self.gae_lambda: float = cfg.train.get("gae_lambda", 0.95)

        # If specified, stop gradient update once KL difference reaches it
        self.target_kl: Optional[float] = cfg.train.target_kl

        # Number of times the collected data is used in gradient update
        self.update_epochs: int = cfg.train.update_epochs

        # Entropy loss coefficient
        self.ent_coef = cfg.train.get("ent_coef", 0.01)

        # Value loss coefficient
        self.vf_coef = cfg.train.get("vf_coef", 0.5)

        # Whether to use running reward scaling
        self.reward_scale_running: bool = cfg.train.reward_scale_running
        
        # Scaling reward with constant
        self.reward_scale_const: float = cfg.train.get("reward_scale_const", 1)
        
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(cfg.env.n_envs)

        # appended by ReinFlow Authors: use base policy to regularize policy. 
        self.use_bc_loss: bool = cfg.train.get("use_bc_loss", False)
        self.bc_loss_type: bool = cfg.train.get("bc_loss_type", None) if self.use_bc_loss else None
        bc_loss_type_candidates=['W2', 'KL', None]
        if not self.bc_loss_type in bc_loss_type_candidates:
            raise ValueError(f"Unsupported bc_loss_type={self.bc_loss_type}. It must be in the following: {bc_loss_type_candidates}")
        self.bc_coeff: float = cfg.train.get("bc_loss_coeff", 0.0)
        
        # appended by ReinFlow Authors:
        self.current_best_reward = np.float32('-inf')
        self.is_best_so_far = False 
        self.total_steps = self.n_steps * self.n_envs # total number of actions
        self.buffer = None
        self.verbose = cfg.train.get('verbose', False)
        self.actor_update_freq = cfg.train.get('actor_update_freq',1)
        self.actor_update_epoch = cfg.train.get('actor_update_epoch',1)
        self.denoising_steps = cfg.get('denoising_steps', 1)  # 1 is for gaussian policy
        self.skip_initial_eval = False 
        
    def visualize_lr(self, cfg):
        steps = []
        critic_lrs = []
        actor_lrs = []
        for step in range(cfg.train.n_train_itr):
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            steps.append(step)
            critic_lrs.append(self.critic_optimizer.param_groups[0]["lr"])
            actor_lrs.append(self.actor_optimizer.param_groups[0]["lr"])
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(steps, actor_lrs, label='actor', color = 'blue')
        plt.legend(loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(steps, critic_lrs,label='critic', color='red')
        plt.legend(loc='upper right')
        lr_save_path = os.path.join(self.logdir, 'test_lr_schedulers.png')
        plt.savefig(lr_save_path)
        log.info(f"learning rate saved to {lr_save_path}")
        plt.close()
        
        if isinstance(self.actor_lr_scheduler, CustomScheduler):
            self.actor_lr_scheduler.reset()
        if isinstance(self.critic_lr_scheduler, CustomScheduler):
            self.critic_lr_scheduler.reset()
        
        self.print_architecture()
    
    def print_architecture(self):
        import os
        arc_path=os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()
        
    def reset_actor_optimizer(self):
        """Not used anywhere currently"""
        new_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=self.cfg.train.actor_lr,
            weight_decay=self.cfg.train.actor_weight_decay,
        )
        new_optimizer.load_state_dict(self.actor_optimizer.state_dict())
        self.actor_optimizer = new_optimizer
        
        new_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=self.cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=self.cfg.train.actor_lr,
            min_lr=self.cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=self.cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        new_scheduler.load_state_dict(self.actor_lr_scheduler.state_dict())
        self.actor_lr_scheduler = new_scheduler
        log.info("Reset actor optimizer")
    
    def update_lr(self):
        self.critic_lr_scheduler.step()
        if self.itr >= self.n_critic_warmup_itr: 
            self.actor_lr_scheduler.step()
        log.info(f"""learning rate updated. actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}, critic_lr={self.critic_optimizer.param_groups[0]["lr"]:.2e}""")
    
    ########################################################################
    # appended by ReinFlow Authors
    def prepare_video_path(self):
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        self.options_venv = [{} for _ in range(self.n_envs)]
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                self.options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )

    def set_model_mode(self):
        # Define train or eval - all envs restart
        if self.skip_initial_eval and self.itr ==0:
            self.eval_mode = False 
        else:
            if self.resume:
                self.eval_mode = True
                self.resume = False
            else:
                self.eval_mode = self.itr % self.val_freq == 0 and not self.force_train
        self.model.eval() if self.eval_mode else self.model.train()
        self.last_itr_eval = self.eval_mode

    def prepare_run(self):
        # Start training loop
        self.timer = Timer()
        self.run_results = []
        self.cnt_train_step = 0
        self.last_itr_eval = False
        self.done_venv = np.zeros((1,self.n_envs))
    
    def reset_env(self, buffer_device='cpu'):
        # set_seed_everywhere(self.seed)
        # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
        if self.reset_at_iteration or self.eval_mode or self.last_itr_eval:
            self.prev_obs_venv = self.reset_env_all(options_venv=self.options_venv)
            self.buffer.firsts_trajs[0] = 1
        else:
            # if done at the end of last iteration, the envs are just reset
            if buffer_device == 'cpu':
                self.buffer.firsts_trajs[0] = self.done_venv
            else:
                self.buffer.firsts_trajs[0] = torch.from_numpy(self.done_venv).float().to(buffer_device)
    
    def save_model(self):
        """
        overload. 
        saves model to disk; no ema recorded. 
        TODO: save ema
        """
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
            "critic_lr_scheduler": self.critic_lr_scheduler.state_dict(),
        }
        
        # always save the last model for resume of training. 
        save_path = os.path.join(self.checkpoint_dir,f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))
        # log.info(f"\n Saved last model at itr {self.itr} to {save_path}\n ")
        
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
    
    def plot_state_trajecories(self): 
        if not self.traj_plotter:
            return 
        if self.itr % self.render_freq == 0 and self.n_render > 0:
                self.traj_plotter(
                    obs_full_trajs=self.obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )
    def resume_training(self):
        log.info(f"Resuming training...")
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        print(f"Recover checkpoint: data={data.keys()}")
        # recover itr, scheduler, sample number
        self.itr = data["itr"]
        self.cnt_train_step = self.itr * self.n_envs * self.act_steps * self.n_steps if 'cnt_train_steps' not in data.keys() else data["cnt_train_steps"]
        self.n_train_itr += self.itr # train for another xx iters. 
        log.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")
        
        # load models
        if "model" in data.keys():
            self.model.load_state_dict(data["model"], strict=True)
            log.info(f"""Loaded full model, including {data["model"].keys()}""")
        elif "policy" in data.keys():
            self.model.actor_ft.policy.load_state_dict(data["policy"], strict=True)
            log.info(f"Loaded policy. Initialize critic and add noise network from scratch.")
        else:
            raise ValueError(f"Your saved checkpoint does not contain keys like 'model' and 'policy'. Please check what was wrong with your model saving functions. ")
        
        log.info(f"Successfully loaded model from path={self.resume_path}")
        
        # load optimizers
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        log.info(f"Successfully loaded optimizers from path={self.resume_path}") 
        
        # load scheduler
        if 'actor_lr_scheduler' in data.keys():
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
            self.critic_lr_scheduler.load_state_dict(data["critic_lr_scheduler"])
            log.info(f"Successfully loaded schedulers from path={self.resume_path}")
        else:
            for _ in range(self.itr): # recover lr schedulers 
                self.actor_lr_scheduler.step()
                self.critic_lr_scheduler.step()
            log.info(f"No schedulers found in path={self.resume_path}. Automatically calibrate the newly initialized scheduler.")
    
    def update_step(self, batch):
        raise NotImplementedError
    
    def agent_update(self):
        # put all the samples in n_steps x n_envs in a line
        obs, samples, returns, values, advantages, logprobs = self.buffer.make_dataset()
        
        # Explained variation of future rewards using value function
        explained_var = self.buffer.get_explained_var(values, returns)
        
        clipfrac_list = []
        
        # generate a random minibatch of data. 
        for update_epoch in range(self.update_epochs):
            kl_change_too_much = False
            indices = torch.randperm(self.total_steps, device=self.device)
            for start in range(0, self.total_steps, self.batch_size):
                end = start + self.batch_size
                minibatch_idx = indices[start:end]

                batch = (
                    {"state": obs[minibatch_idx]},
                    samples[minibatch_idx],
                    returns[minibatch_idx],
                    values[minibatch_idx],
                    advantages[minibatch_idx],
                    logprobs[minibatch_idx]
                )

                # minibatch gradient descent
                pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, std = self.update_step(batch) 
                
                loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_coeff
                
                clipfrac_list += [clipfrac]
                
                # update policy and critic
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                loss.backward()
                
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm : torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Stop gradient update if KL difference reaches target
                if self.target_kl and approx_kl > self.target_kl:
                    kl_change_too_much = True
                    break
            if kl_change_too_much:
                break
        clip_frac = np.mean(clipfrac_list)
        self.train_ret_dict = {
                "loss": loss,
                "pg loss": pg_loss,
                "value loss": v_loss,
                "entropy_loss": entropy_loss,
                "std": std,
                "approx kl": approx_kl,
                "ratio": ratio,
                "clip_frac": clip_frac,
                "explained variance": explained_var,
            }

    def log(self, train_prt_str_additional="", train_log_dict_additional={}):
        '''
        train_prt_str_additional: str, additional information in training that will be printed to log console that is not included in train_prt_str_basic
        train_log_dict_additional: dict, additional information in training that will be logged to wandb that is not included in train_log_dict_basic
        '''
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'

        self.run_results.append(
            {
                "itr": self.itr,
                "step": self.cnt_train_step,
            }
        )
        if self.save_trajs:
            self.run_results[-1]["self.obs_full_trajs"] = self.obs_full_trajs
            self.run_results[-1]["self.obs_trajs"] = self.obs_trajs
            self.run_results[-1]["action_trajs"] = self.samples_trajs
            self.run_results[-1]["self.reward_trajs"] = self.reward_trajs
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                # Updated evaluation log with avg ± std formatting
                log.info(create_bordered_text(
                    f"{BOLDSTART}Evaluation at itr {self.itr}{BOLDEND}:\n"
                    f"Model: {self.model.__class__.__name__}\n"
                    f"Environment: {self.env_name} x {self.n_envs}\n"
                    f"Num denoising steps: {self.denoising_steps}\n"
                    f"Seed: {self.seed}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}%\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Best Reward (per action): {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.2e}"
                ))
                eval_dict={
                            "eval/success rate": self.buffer.success_rate,
                            "eval/avg episode reward": self.buffer.avg_episode_reward,
                            "eval/avg best reward": self.buffer.avg_best_reward,
                            "eval/avg episode length": self.buffer.avg_episode_length,
                            "eval/num episode": self.buffer.num_episode_finished,
                            "eval/std success rate": self.buffer.std_success_rate,
                            "eval/std episode reward": self.buffer.std_episode_reward,
                            "eval/std best reward": self.buffer.std_best_reward,
                            "eval/std episode length": self.buffer.std_episode_length,
                    }
                # convert everything to floating points
                for key, value in eval_dict.items():
                    if isinstance(value, torch.Tensor):
                        eval_dict[key]=value.item()
                self.run_results[-1].update(eval_dict)
                if self.use_wandb:
                    wandb.log(
                        data=eval_dict,
                        step=self.itr,
                        commit=False,
                    )
                
                if self.current_best_reward < self.buffer.avg_episode_reward:
                    self.current_best_reward = self.buffer.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                # Updated training log with avg ± std formatting
                train_prt_str_basic = (
                    f"itr {self.itr} | Total Step {self.cnt_train_step / 1e6:4.3f} M | Time: {time:8.3f}\n"
                    f"Env: {self.env_name} x {self.n_envs}\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}% \n"
                    f"Avg Best Reward: {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                    f"Critic lr: {self.critic_optimizer.param_groups[0]['lr']:.2e}\n"
                )
                formatted_items = [f"{key}: {value:.3e}" for key, value in self.train_ret_dict.items()]
                num_items_per_row = 10
                for i in range(0, len(formatted_items), num_items_per_row):
                    train_prt_str_basic += " | ".join(formatted_items[i:i+num_items_per_row]) + "\n"
                log.info(train_prt_str_basic + train_prt_str_additional)
                
                # upload to wandb
                train_log_dict_basic = {
                    "train/total env step": self.cnt_train_step,
                    "train/success rate": self.buffer.success_rate,
                    "train/avg episode reward": self.buffer.avg_episode_reward,
                    "train/avg episode length": self.buffer.avg_episode_length,
                    "train/num episode": self.buffer.num_episode_finished,                    
                    "train/std success rate": self.buffer.std_success_rate,
                    "train/avg best reward": self.buffer.avg_best_reward,
                    "train/std episode reward": self.buffer.std_episode_reward,
                    "train/std best reward": self.buffer.std_best_reward,
                    "train/std episode length": self.buffer.std_episode_length,
                    "train/actor lr": self.actor_optimizer.param_groups[0]["lr"],
                    "train/critic lr": self.critic_optimizer.param_groups[0]["lr"],
                }
                loss_dict = {"loss/"+key: value for key, value in self.train_ret_dict.items()}
                train_log_dict_basic.update(loss_dict)
                train_log_dict = {**train_log_dict_basic, **(train_log_dict_additional or {})}
                # convert everything to floating points
                for key, value in train_log_dict.items():
                    if isinstance(value, torch.Tensor):
                        train_log_dict[key]=value.item()
                self.run_results[-1].update(train_log_dict)
                
                # Log training metrics to WandB
                if self.use_wandb:
                    wandb.log(
                        data=train_log_dict,
                        step=self.itr,
                        commit=True,
                    )
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)