"""
Parent pre-training agent class.
"""
import os
import random
import numpy as np
from omegaconf import OmegaConf
from util.scheduler import CosineAnnealingWarmupRestarts
import torch
import hydra
import logging
import wandb
from copy import deepcopy
log = logging.getLogger(__name__)
from util.timer import Timer
from tqdm import tqdm
from agent.pretrain.utils  import batch_to_device
from util.dirs import REINFLOW_DATA_DIR
from env.gym_utils import make_async
# to test in mujoco simulator
class EnvConfig:
    def __init__(self, n_envs, 
                 name, 
                 max_episode_steps, 
                 reset_at_iteration, 
                 save_video, 
                 use_image_obs, 
                 best_reward_threshold_for_success,
                 wrappers, 
                 n_steps, 
                 render, 
                 render_num,
                 robomimic_env_cfg_path:str=None,
                 shape_meta=None):
        self.n_envs = n_envs
        self.name = name
        self.max_episode_steps = max_episode_steps
        self.reset_at_iteration = reset_at_iteration
        self.save_video = save_video
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.wrappers = wrappers
        self.n_steps = n_steps
        self.render_num = render_num 
        self.use_image_obs = use_image_obs  # Change to True if using image observations in the environment
        self.render = render
        self.specific={}  # not implemented for furniture right now.
        self.robomimic_env_cfg_path=robomimic_env_cfg_path
        self.shape_meta=shape_meta

class PreTrainAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        
        # verbose
        self.verbose_train=False
        self.verbose_loss= False
        self.verbose_test= False
        self.test_model_type='ema' # 'original'
        self.test_log_all = False
        self.only_test=False
        
        # Seed
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if self.use_wandb:
            # Check if offline mode is enabled (e.g., via config or environment variable)
            offline_mode = cfg.wandb.get("offline_mode", False)  # Add this to your config if desired
            if offline_mode:
                wandb_dir = cfg.wandb.get("dir", "./wandb_offline")  # Local directory for offline logs
            else:
                wandb_dir = cfg.wandb.get("dir", "./wandb")
            os.makedirs(wandb_dir, exist_ok=True)  # Ensure the directory exists
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="offline" if offline_mode else "online",  # Switch to offline mode
                dir=wandb_dir,  # Specify local directory for offline logs
            )
            # Get the exact subfolder for this run
            run_id = wandb.run.id  # Unique ID for the run
            run_dir = wandb.run.dir  # Full path to the run's directory
            if offline_mode:
                log.info(f"Wandb running in offline mode. Logs will be saved to {os.path.dirname(run_dir)}")
                log.info(f"Run ID: {run_id}")
            else:
                log.info(f"Wandb running online. Run ID: {run_id}")
        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.log_all = cfg.train.get("log_all", True)
        
        # Build model
        self.model = hydra.utils.instantiate(cfg.model)
        self.ema = EMA(cfg.ema)
        self.ema_model = deepcopy(self.model)
        self.print_architecture()
        
        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.epoch_start_ema = cfg.train.get("epoch_start_ema", 20)
        self.update_ema_freq = cfg.train.get("update_ema_freq", 10)
        self.val_freq = cfg.train.get("val_freq", 100)
        
        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        print(f"dataset_train={len(self.dataset_train)}")
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
            drop_last=True # revised by ReinFlow Authors when debugging flow-matching shortcut. 
        )
        self.dataloader_val = None
        if "train_split" in cfg.train and cfg.train.train_split < 1:
            val_indices = self.dataset_train.set_train_val_split(cfg.train.train_split)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=True,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
                drop_last=True # revised by ReinFlow Authors when debugging flow-matching shortcut. 
            )
        
        # optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.schedule_lr_each_grad_step = cfg.train.get("schedule_lr_each_grad_step", False)
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.learning_rate,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.reset_parameters()

        # resume
        self.epoch = 1
        self.first_epoch = 0
        # when resume from checkpoint, just reduce the train.n_epochs in your configuration file by the amount of epochs you have trained. 
        # our code will automatically restore the checkpoint for the model and the optimizer and the learning rate scheduler and train for another `train.n_epochs` steps. 
        self.resume_path = cfg.get('base_policy_path', None)
        self.resume = self.resume_path is not None
        if self.resume:
            log.info(f"Resuming from {self.resume_path}")
            self.load(epoch=None, custom_path=self.resume_path)
            log.info(f"Resume complete.")

        # Testing in mujoco 
        self.test_in_mujoco = cfg.get('test_in_mujoco', False) # in openai gym envs, we test the training performance in mujoco to save the model with the highest reward
        self.test_freq = cfg.train.get('test_freq', self.n_epochs -1)
        log.info(f"test_in_mujoco=={self.test_in_mujoco}")
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.test_each_step = False
        
        if self.test_in_mujoco:
            self.test_denoising_steps=20 #to be overloaded
            self.test_model_type='ema'   # original
            
            self.best_episode_reward=0.0
            self.success_rate = 0.0
            self.avg_episode_reward = 0.0
            self.avg_episode_reward_std=0.0
            self.avg_episode_length=0.0
            self.avg_episode_length_std=0.0
            self.avg_best_reward=0.0
            self.avg_best_reward_std=0.0
            ############################################ test in mujoco simulator #########################################
            # Create an instance of EnvConfig with the values from the annotated environment
            self.env_name = cfg.env
            env_type = cfg.get("env_type", None)
            act_steps = cfg.horizon_steps # for now
            normalization_path=cfg.get('normalization_path', None) # )
            if not normalization_path:
                raise ValueError(f"Hey you must specify your normalization path if you wish to evaluate periodically during pretraining, but I can't find it in your configuration! Is {os.path.join(REINFLOW_DATA_DIR,'gym',cfg.env,'normalization.npz')} the correct path I guess? Also, make sure to secure that your normalization file truly correspond to your pretraining data, other wise there will be a significant mismatch in preformance. ")
            if cfg.env_suite=='gym':
                env_max_episode_steps=1_000
                rollout_n_steps=5_00
                n_eval_envs=40
                best_reward_threshold_for_success=3.0
                robomimic_env_cfg_path=None
                shape_meta=None
                use_image_obs=False
                wrappers={
                        "mujoco_locomotion_lowdim": {
                            "normalization_path": normalization_path
                        },
                        "multi_step": {
                            "n_obs_steps": cfg.cond_steps,
                            "n_action_steps": act_steps,
                            "max_episode_steps": env_max_episode_steps,
                            "reset_within_step": True
                        }
                    }
            elif cfg.env_suite=='robomimic':
                env_max_episode_steps=cfg.eval_env.max_episode_steps
                rollout_n_steps=cfg.eval_env.n_steps
                n_eval_envs=cfg.eval_env.n_envs
                best_reward_threshold_for_success=cfg.eval_env.best_reward_threshold_for_success
                robomimic_env_cfg_path=cfg.robomimic_env_cfg_path
                shape_meta=cfg.get('shape_meta', None)
                use_image_obs=cfg.eval_env.get("use_image_obs", False),
                wrappers=cfg.eval_env.wrappers
            else:
                raise NotImplementedError(f"Sorry about that, we have not yet implemented evaluation for cfg.env_suite={cfg.env_suite} environment with MuJoCo simulator during pre-training. Coming soon!")
            self.env_config = EnvConfig(
                n_envs=n_eval_envs,
                name=cfg.env,
                max_episode_steps=env_max_episode_steps,
                reset_at_iteration=False,
                save_video=False,                               # Change to True if needed
                use_image_obs = use_image_obs,
                best_reward_threshold_for_success=best_reward_threshold_for_success,
                wrappers=wrappers,
                n_steps=rollout_n_steps,
                render=False,
                render_num=0,
                robomimic_env_cfg_path=robomimic_env_cfg_path,
                shape_meta=shape_meta
            )
            # create mujoco simulator env
            self.venv = make_async(
                self.env_name,
                env_type=env_type,
                num_envs=self.env_config.n_envs,
                asynchronous=True,
                max_episode_steps=self.env_config.max_episode_steps,
                wrappers=self.env_config.wrappers,
                robomimic_env_cfg_path=self.env_config.robomimic_env_cfg_path,
                shape_meta=self.env_config.shape_meta,
                use_image_obs=self.env_config.use_image_obs,
                render=self.env_config.render,
                render_offscreen=self.env_config.save_video,
                obs_dim=cfg.obs_dim,
                action_dim=cfg.action_dim,
                **cfg.eval_env.specific if "specific" in cfg.env else {},
            )
            if not env_type == "furniture":
                self.venv.seed(
                    [self.seed + i for i in range(self.env_config.n_envs)]
                )
            self.n_envs = self.env_config.n_envs
            self.n_cond_step = cfg.cond_steps
            self.obs_dim = cfg.obs_dim
            self.action_dim = cfg.action_dim
            self.act_steps = cfg.horizon_steps # act_steps
            self.horizon_steps = cfg.horizon_steps
            self.max_episode_steps = self.env_config.max_episode_steps
            self.reset_at_iteration = self.env_config.reset_at_iteration
            self.furniture_sparse_reward= (
                self.env_config.specific.get("sparse_reward", False)
                if "specific" in cfg.env
                else False
            )
            # Now, replace references to cfg and its parameters with eval_config
            self.n_steps = self.env_config.n_steps
            self.best_reward_threshold_for_success = self.env_config.best_reward_threshold_for_success
            # rendering
            self.n_render = self.env_config.render_num
            self.render_video = self.env_config.save_video  # Assuming you want to use the save_video from self.env_config
            assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
            assert not (
                self.n_render <= 0 and self.render_video
            ), "Need to set n_render > 0 if saving video"

        self.print_architecture()
    
    def print_architecture(self):
        import os
        arc_path=os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()

    # for debugging
    def test_lr_scheduler(self):
        lrs=[]
        steps=[]
        for epoch in range(self.first_epoch, self.first_epoch + self.n_epochs):
            self.lr_scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])
            steps.append(epoch)
        import matplotlib.pyplot as plt 
        plt.plot(steps, lrs)
        figpath = f"agent/pretrain/test_resume.png"
        plt.savefig(figpath)
        print(f"saved lr resume test result to {figpath}")
        exit()

    def get_loss(self, batch_data):
        '''for training and validation on fixed dataset'''
        raise NotImplementedError

    def inference(self, cond:dict):
        '''for evaluation in sim'''
        raise NotImplementedError
    
    def run(self):
        print(f"dataloader_train={len(self.dataloader_train)}")
        self.test() # see the initialization or resumed performance.
        if self.only_test:
            exit()
        
        # this is for diffusion and reflow models. 
        timer = Timer()
        cnt_batch = 0
        log.info(f"self.epoch={self.epoch}, begin training.")
        for epoch in tqdm(range(self.first_epoch, self.first_epoch + self.n_epochs)):
            self.model.train()
            # train
            loss_train_epoch = []
            steps_per_epoch = len(self.dataloader_train)
            for step, batch_train in tqdm(enumerate(self.dataloader_train), desc=f'total steps={steps_per_epoch}') \
                if self.verbose_train else enumerate(self.dataloader_train):
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)
                
                self.optimizer.zero_grad()
                
                loss_train = self.get_loss(batch_train)
                
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                if self.verbose_loss: 
                    print(f"epoch: {epoch}/{self.first_epoch + self.n_epochs}={epoch/(self.n_epochs-self.first_epoch)*100:2.2f}%, steps: {step}, loss: {loss_train.item():3.4}", end="\r")

                self.optimizer.step()
                if self.schedule_lr_each_grad_step:
                    self.lr_scheduler.step()
                
                # update ema
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            loss_train = np.mean(loss_train_epoch)

            # validate
            with torch.no_grad():
                loss_val_epoch = []
                # for RL, self.dataloader_val is None. So you just skip this part. 
                if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                    self.model.eval()
                    for batch_val in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch_val = batch_to_device(batch_val)
                        with torch.no_grad:
                            loss_val = self.get_loss(batch_val)
                            loss_val_epoch.append(loss_val.item())
                    self.model.train()
                loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            if not self.schedule_lr_each_grad_step:
                self.lr_scheduler.step()
            
            # always save the last checkpoint for resume 
            self.save_last_model()
            
            # save model # default is 100 by pre_diffusion_mlp.yaml
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
                        
            # test in mujoco simulator
            if self.test_in_mujoco and self.epoch % self.test_freq == 0:
                self.test()
            
            # log testing info
            self.log(epoch, loss_train, loss_val, timer)
            self.epoch += 1
    

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def save_model(self):
        """
        saves model and ema to disk;
        """
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}\n")

    def save_best_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best.pt")
        torch.save(data, savepath)
        log.info(f"Saved the best model to {savepath}\t It has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}.")

    def save_best_ema_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"best_ema.pt")
        torch.save(data, savepath)
        log.info(f"Saved the best EMA model to {savepath}, which has highest self.avg_episode_reward: {self.best_episode_reward:8.2f}.")
        
    def save_last_model(self):
        '''for resume purpose'''
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"last.pt")
        torch.save(data, savepath)
        # log.info(f"Saved the last model to {savepath}")
    
    def load(self, epoch, custom_path=None):
        """
        loads model and ema from disk
        """
        if custom_path:
            loadpath = os.path.join(custom_path)
        else:
            loadpath = os.path.join(self.checkpoint_dir, f"state_{epoch}.pt")
        
        data = torch.load(loadpath, weights_only=True)
        
        # when resume from checkpoint, just reduce the train.n_epochs in your configuration file by the amount of epochs you have trained. 
        # our code will automatically restore the checkpoint for the model and the optimizer and the learning rate scheduler and train for another `train.n_epochs` steps. 
        self.epoch = data["epoch"]
        self.first_epoch = self.epoch+1
        log.info(f"Resume from self.epoch={self.epoch}. Will start from self.first_epoch={self.first_epoch} and train for another {self.n_epochs} epochs. ")
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
    
    # log loss
    def log(self, epoch, loss_train, loss_val, timer):
        if self.epoch % self.log_freq == 0:  # default is every step. 
            log.info(f"{self.epoch}: train loss {loss_train:8.4f} | t:{timer():8.4f}")
            if self.use_wandb:
                if loss_val is not None:
                    wandb.log(
                        {"loss - val": loss_val}, step=self.epoch, commit=False
                    )
                if not self.test_in_mujoco:
                    wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                },
                                step=self.epoch,
                                commit=True,
                            )
                else:
                    if self.log_all:
                        wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "best_reward": self.best_episode_reward,
                                "success_rate": self.success_rate,
                                "avg_episode_reward": self.avg_episode_reward,
                                "avg_episode_reward_std": self.avg_episode_reward_std,
                                "avg_episode_length": self.avg_episode_length,
                                "avg_best_reward": self.avg_best_reward,
                                "avg_best_reward_std": self.avg_best_reward_std,
                                },
                                step=self.epoch,
                                commit=True,
                            )
                    else:
                        wandb.log(
                                {
                                "loss - train": loss_train,
                                "lr": self.optimizer.param_groups[0]["lr"],
                                "best_reward": self.best_episode_reward,
                                # "success_rate": self.success_rate,
                                # "avg_episode_reward": self.avg_episode_reward,
                                # "avg_episode_reward_std": self.avg_episode_reward_std
                                },
                                step=self.epoch,
                                commit=True,
                            )
        # count    
        log.info(f"epoch={epoch} , self.first_epoch={self.first_epoch}, last epoch={self.first_epoch + self.n_epochs}, loss_train={loss_train:.4f}")



    ##################### test in mujoco simulator for openai gym with state inputs #####################
    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        # convert to OrderedDict if obs_venv is a list of dict
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
    
    def test(self):
        if not self.test_in_mujoco:
            return
        log.info(f"Evaluating {self.model.__class__.__name__} in environment {self.env_name} with denoising steps = {self.test_denoising_steps}")
        
        log_all= self.test_log_all
        timer = Timer()
        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        
        self.avg_episode_length = 0.0 
        # Collect a set of trajectories from env
        for step in tqdm(range(self.n_steps)) if self.verbose_test else range(self.n_steps):
            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                
                # different models differs here.
                samples = self.inference(
                    cond=cond
                )
                
                # here, samples is a namedtuple of class `Sample(trajetories, chains)` trajectories is a single action forecast, and chains is the whole generation. 
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]

            # Apply multi-step action
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv

            # update for next step
            prev_obs_venv = obs_venv

        # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if (
                self.furniture_sparse_reward
            ):  # only for furniture tasks, where reward only occurs in one env step
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_episode_reward_std = np.std(episode_reward)
            self.success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            if log_all:
                self.avg_best_reward = np.mean(episode_best_reward)
                self.avg_best_reward_std = np.std(episode_best_reward)
                episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
                traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                traj_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
                self.avg_episode_length=traj_length
                self.avg_episode_length_std=traj_std
        else:
            self.avg_episode_reward = 0
            self.avg_episode_reward_std=0.0
            self.success_rate = 0
            if log_all:
                episode_reward = np.array([])
                num_episode_finished = 0
                self.avg_best_reward = 0
                self.avg_best_reward_std=0.0
                episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
                traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
                traj_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
                self.avg_episode_length=traj_length
                self.avg_episode_length_std=traj_std
            
            log.info("[WARNING] No episode completed within the iteration!")

        if self.avg_episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.avg_episode_reward
            self.save_best_model()
            log.info(f"Current Best model saved at epoch {self.epoch}")

        time = timer()
        if log_all:
            log.info(
                    f"eval: success rate {self.success_rate:8.4f} | avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f} | avg_episode_length {self.avg_episode_length:4.2f}±{self.avg_episode_length_std:4.2f} | num episode {num_episode_finished:4d} | avg best reward {self.avg_best_reward:8.1f}±{self.avg_best_reward_std:2.1f} |"
                )
        else:
            log.info(
                    f"eval: success rate {self.success_rate*100:2.2f}%| avg episode reward {self.avg_episode_reward:8.1f}±{self.avg_episode_reward_std:2.1f}"
                )
        if log_all:
            np.savez(
                self.result_path,
                num_episode=num_episode_finished,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_best_reward=self.avg_best_reward,
                time=time,
            )
        else:
            np.savez(
                self.result_path,
                eval_success_rate=self.success_rate,
                eval_episode_reward=self.avg_episode_reward,
                eval_episode_reward_std=self.avg_episode_reward_std,
                time=time,
            )
class EMA:
    """
    Exponential moving average
    """
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.decay
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new