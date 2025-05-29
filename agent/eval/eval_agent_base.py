




"""
Parent eval agent class with state input, for openai-gym environment
"""
import os
import numpy as np
import torch
import hydra
import logging
import random
from tqdm import tqdm as tqdm
log = logging.getLogger(__name__)
from env.gym_utils import make_async
from omegaconf import OmegaConf
import torch.nn as nn
import os
from agent.eval.visualize.utils import read_eval_statistics
from util.dirs import REINFLOW_DIR 
class EvalAgent:
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.device = cfg.device
        self.base_policy_path = cfg.base_policy_path
        self.eval_log_dir = cfg.get('eval_log_dir', None)
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        ############ could be overload #############
        self.record_video = False
        self.record_env_index = -1
        self.render_onscreen = False
        self.denoising_steps = None
        self.denoising_steps_trained = None
        self.plot_scale = cfg.get("plot_scale", "semilogx")  # Default to semilogx, can be "standard" or "semilogx"
        self.plot_scale_options=['standard, semilogx']
        # if self.plot_scale not in self.plot_scale_options:
        #     raise ValueError(f"plot scale must be one of {self.plot_scale_options}, but received {self.plot_scale}!")
        ############################################
        
        # Make vectorized env
        self.env_name = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )
        
        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        
        # Eval params
        self.n_steps = cfg.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.render_num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        
    
    def load_model_for_eval(self):
        data = torch.load(self.base_policy_path, weights_only=True, map_location=self.device)
        self.model: nn.Module        
        print(f"loading model...")
        if self.load_ema:
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=False)
                else:
                    actor_policy_state_dict = {key.replace('actor_ft.policy.', 'network.'): value 
                                        for key, value in data["ema"].items() 
                                        if key.startswith('actor_ft.policy.')}
                    if actor_policy_state_dict == {}:
                        raise ValueError(f"No parameter starting with actor_ft.policy in ={data['ema'].keys()}")
                    self.model.load_state_dict(actor_policy_state_dict, strict=False)
            else:
                raise ValueError(f"You set self.load_ema={self.load_ema}, but your state dictionary does not contain key: ema. It only contains keys: {data.keys()}")
            log.info(f"Loaded EMA model dict from {self.base_policy_path}")
        else:
            if 'policy' in data.keys():
                self.model.load_state_dict(data["policy"], strict=True)
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=True)
            elif 'model' in data.keys():
                actor_policy_state_dict = {} 
                for key, value in data["model"].items():
                    if key.startswith('actor_ft.mlp_logvar') or key.startswith('actor_ft.logvar') or key.startswith('actor_ft.explore_noise_net.'):
                        continue
                    if key.startswith('actor_ft.policy.'):
                        actor_policy_state_dict[key.replace('actor_ft.policy.', 'network.')] = value
                    elif key.startswith('actor_ft.'):
                        actor_policy_state_dict[key.replace('actor_ft.', 'network.')] = value
                if actor_policy_state_dict == {}:
                    raise ValueError(f"No parameter starting with actor_ft.policy or actor_ft. in ={data['model'].keys()}")
                self.model.load_state_dict(actor_policy_state_dict, strict=True)
            else:
                raise ValueError(f"Your state dictionary is not correct, it does not contain keys: policy or model. It only contains keys: {data.keys()}")
            log.info(f"Loaded model dict from {self.base_policy_path}")

    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
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
    
    
    def run(self):
        if self.render_onscreen and self.n_envs > 1:
            raise ValueError(f"Cannot render on screen with more than one parallel envs. self.render_onscreen={self.render_onscreen}, cfg.env.n_envs={self.n_envs}")
        if self.eval_log_dir is None:
            self.eval_log_dir = f'visualize/{self.model.__class__.__name__}/{self.env_name}/{self.current_time()}/'
        os.makedirs(self.eval_log_dir, exist_ok=True)
        cfg_path = os.path.join(self.eval_log_dir, "cfg.yaml")
        with open(cfg_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        print(f"Configuration saved to {cfg_path}")
        
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        
        self.load_model_for_eval()
        denoising_steps_set = self.denoising_steps
        
        # Lists to store the results
        num_denoising_steps_list = []
        avg_single_step_freq_list = []
        avg_single_step_freq_std_list = []
        avg_single_step_duration_list = []
        avg_single_step_duration_std_list = []
        avg_traj_length_list = []
        avg_traj_length_std_list = []  # Added
        avg_episode_reward_list = []
        avg_episode_reward_std_list = []
        avg_best_reward_list = []
        avg_best_reward_std_list = []
        success_rate_list = []
        success_rate_std_list = []  # Added
        num_episodes_finished_list = []
        
        for num_denoising_steps in denoising_steps_set:
            self.venv.reset()
            result = self.single_run(num_denoising_steps, options_venv)
            
            num_denoising_steps, avg_single_step_freq, avg_single_step_freq_std, \
                avg_single_step_duration, avg_single_step_duration_std, \
                avg_traj_length, avg_traj_length_std, \
                avg_episode_reward, avg_episode_reward_std, \
                avg_best_reward, avg_best_reward_std, \
                num_episodes_finished, success_rate, success_rate_std = result
            
            num_denoising_steps_list.append(num_denoising_steps)
            avg_single_step_freq_list.append(avg_single_step_freq)
            avg_single_step_freq_std_list.append(avg_single_step_freq_std)
            avg_single_step_duration_list.append(avg_single_step_duration)
            avg_single_step_duration_std_list.append(avg_single_step_duration_std)
            avg_traj_length_list.append(avg_traj_length)
            avg_traj_length_std_list.append(avg_traj_length_std)  # Added
            avg_episode_reward_list.append(avg_episode_reward)
            avg_episode_reward_std_list.append(avg_episode_reward_std)
            avg_best_reward_list.append(avg_best_reward)
            avg_best_reward_std_list.append(avg_best_reward_std)
            success_rate_list.append(success_rate)
            success_rate_std_list.append(success_rate_std)  # Added
            num_episodes_finished_list.append(num_episodes_finished)
        
        # Save evaluation statistics as an npz
        dtype = [
            ('num_denoising_steps', int),
            ('avg_single_step_freq', float),
            ('avg_single_step_freq_std', float),
            ('avg_single_step_duration', float),
            ('avg_single_step_duration_std', float),
            ('avg_traj_length', float),
            ('avg_traj_length_std', float),  # Added
            ('avg_episode_reward', float),
            ('avg_episode_reward_std', float),
            ('avg_best_reward', float),
            ('avg_best_reward_std', float),
            ('success_rate', float),
            ('success_rate_std', float),  # Added
            ('num_episodes_finished', int)
        ]

        data = np.zeros(len(num_denoising_steps_list), dtype=dtype)
        data['num_denoising_steps'] = num_denoising_steps_list
        data['avg_single_step_freq'] = avg_single_step_freq_list
        data['avg_single_step_freq_std'] = avg_single_step_freq_std_list
        data['avg_single_step_duration'] = avg_single_step_duration_list
        data['avg_single_step_duration_std'] = avg_single_step_duration_std_list
        data['avg_traj_length'] = avg_traj_length_list
        data['avg_traj_length_std'] = avg_traj_length_std_list  # Added
        data['avg_episode_reward'] = avg_episode_reward_list
        data['avg_episode_reward_std'] = avg_episode_reward_std_list
        data['avg_best_reward'] = avg_best_reward_list
        data['avg_best_reward_std'] = avg_best_reward_std_list
        data['success_rate'] = success_rate_list
        data['success_rate_std'] = success_rate_std_list  # Added
        data['num_episodes_finished'] = num_episodes_finished_list

        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')
        np.savez(eval_statistics_path, data=data)
        
        statistics = read_eval_statistics(npz_file_path=eval_statistics_path)
        self.plot_eval_statistics(statistics, self.eval_log_dir)

    def single_run(self, num_denoising_steps, options_venv):
        import cv2
        self.video_writer = None
        if self.record_video:
            frame_width = 640  # Default, can be overridden
            frame_height = 480
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.eval_log_dir, f'{self.model.__class__.__name__}_{self.env_name}_step{num_denoising_steps}.mp4')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps"
        
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        single_step_duration_list = np.zeros(self.n_steps)
        
        # Log message can remain as is
        log.info(f"Evaluating {self.model.__class__.__name__} model in {self.env_name} environment with {num_denoising_steps} step(s).")
        
        for step in tqdm(range(self.n_steps), dynamic_ncols=True, desc=f'{num_denoising_steps} step(s):'):
            with torch.no_grad():
                # Generic observation handling
                if hasattr(self, 'obs_dims'):  # For image-based agents
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                else:  # For state-based agents
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                samples, single_step_duration= self.infer(cond, num_denoising_steps)
                
                single_step_duration_list[step] = single_step_duration
                
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]
            
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            if self.render_onscreen:
                self.venv.render(mode='human')
            
            if self.record_video:
                frame_tuple = self.venv.render(mode='rgb_array', height=frame_height, width=frame_width)
                if self.video_writer is not None:
                    frame = frame_tuple[self.record_env_index]
                    cv2.putText(frame, self.video_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.video_writer.write(frame)
            
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            prev_obs_venv = obs_venv
        
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"video saved to {video_path}")

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
            num_episodes_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            success_rate_std = np.std(episode_best_reward >= self.best_reward_threshold_for_success)
        else:
            episode_reward = np.array([])
            num_episodes_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            success_rate_std = 0  # Added
            log.info("[WARNING] No episode completed within the iteration!")
        
        episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
        avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        avg_traj_length_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
        
        avg_single_step_duration = single_step_duration_list.mean()
        avg_single_step_duration_std = single_step_duration_list.std()
        single_step_frequency_list = 1 / single_step_duration_list
        avg_single_step_freq = single_step_frequency_list.mean()
        avg_single_step_freq_std = single_step_frequency_list.std()
        
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'
        log.info(
            f"""
            #############################################################
            {BOLDSTART}Evaluation{BOLDEND}
            Model:               {self.model.__class__.__name__}
            Environment:         {self.env_name} x {self.n_envs}
            denoising steps:     {num_denoising_steps}
            
            success_rate:        {success_rate*100: 8.3f} % ± {success_rate_std*100:8.3f} %
            avg_episode_reward:  {avg_episode_reward:8.1f} ± {avg_episode_reward_std:2.1f}
            
            
            avg_single_step_freq:{avg_single_step_freq:3.1f} ± {avg_single_step_freq_std:3.1f} HZ
            
            avg_traj_length:     {avg_traj_length:3.1f} ± {avg_traj_length_std:3.1f} steps
            avg_best_reward:     {avg_best_reward:8.1f} ± {avg_best_reward_std:2.1f}
            num_episode:         {num_episodes_finished:4d}
            #############################################################
            """
        )
        
        return num_denoising_steps, \
            avg_single_step_freq, avg_single_step_freq_std, \
            avg_single_step_duration, avg_single_step_duration_std, \
            avg_traj_length, avg_traj_length_std, \
            avg_episode_reward, avg_episode_reward_std, \
            avg_best_reward, avg_best_reward_std, \
            num_episodes_finished, success_rate, success_rate_std
    
    def current_time(self):
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
        return formatted_time
    
    def plot_eval_statistics(self, eval_statistics, log_dir: str):
        num_denoising_steps_list, \
            avg_single_step_freq_list, avg_single_step_freq_std_list, \
            avg_single_step_duration_list, avg_single_step_duration_std_list, \
            avg_traj_length_list, avg_traj_length_std_list, \
            avg_episode_reward_list, avg_episode_reward_std_list, \
            avg_best_reward_list, avg_best_reward_std_list, \
            success_rate_list, success_rate_std_list, \
            num_episodes_finished_list = eval_statistics
        
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 8))
        
        # Define plotting function based on self.plot_scale
        plot_func = plt.semilogx if self.plot_scale == "semilogx" else plt.plot

         # Plot success rate
        plt.subplot(2, 3, 1)
        plot_func(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.fill_between(num_denoising_steps_list,
                        [sr - std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        [sr + std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        color='y', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()


        # Plot average episode reward
        plt.subplot(2, 3, 2)
        plot_func(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
        plt.fill_between(num_denoising_steps_list,
                        [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        color='b', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 5)
        plot_func(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.fill_between(num_denoising_steps_list,
                        [avg_traj - std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        [avg_traj + std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        color='r', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Trajectory Length')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Trajectory Length')
        plt.grid(True)
        plt.legend()

        # Plot inference duration
        plt.subplot(2, 3, 3)
        plot_func(num_denoising_steps_list, avg_single_step_duration_list, marker='o', label='Time', color='purple')
        plt.fill_between(num_denoising_steps_list,
                        [duration - std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        [duration + std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        color='purple', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Duration')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Time (s)')
        plt.grid(True)
        plt.legend()
        
        
        # Plot average best reward
        plt.subplot(2, 3, 4)
        plot_func(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
        plt.fill_between(num_denoising_steps_list,
                        [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        color='g', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

       
        # Plot inference frequency
        plt.subplot(2, 3, 6)
        plot_func(num_denoising_steps_list, avg_single_step_freq_list, marker='o', label='Frequency', color='brown')
        plt.fill_between(num_denoising_steps_list,
                        [freq - std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        [freq + std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        color='brown', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Frequency')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Frequency (Hz)')
        plt.grid(True)
        plt.legend()

        plt.suptitle(f"{self.model.__class__.__name__}, {self.env_name}\nsteps = {', '.join(map(str, num_denoising_steps_list))}", fontsize=25)
        plt.tight_layout()
        
        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')


        fig_path = os.path.join(REINFLOW_DIR, log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"Finished evaluating {self.model.__class__.__name__} in environment {self.env_name}.\n\
            Base_policy_path: {os.path.join(REINFLOW_DIR,self.base_policy_path)}\n\
            Figure saved to {fig_path}\n\
            Evaluation statistics saved to  {eval_statistics_path}")
        plt.close()