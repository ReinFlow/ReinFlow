# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Launcher for all experiments. Download pre-training data, normalization statistics, and pre-trained checkpoints if needed.
Revised by ReinFlow Authors to accomodate resume training and fixing the kitchen tasks import error.
"""
# clear python cache automatically.
from util.clear_pycache import clean_pycache
from util.dirs import REINFLOW_DIR
clean_pycache(directory=REINFLOW_DIR)

# register kitchen tasks in advance. prevent env not found error. 
import gym
import d4rl.gym_mujoco

import gc
gc.collect()

import os
import sys
import logging
import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "cfg"),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)
    
    # ReinFlow Authors: Set rendering backend from config file.
    sim_device = cfg.get('sim_device', None)
    if sim_device is not None:
        # Extract numeric device index if cuda: prefix is present
        if isinstance(sim_device, str) and sim_device.startswith('cuda:'):
            try:
                sim_device = int(sim_device.split('cuda:')[1])
            except ValueError:
                raise ValueError(f"Invalid sim_device format: {sim_device}. Expected 'cuda:<number>' or a numeric index.")
        elif isinstance(sim_device, (int, str)):
            try:
                sim_device = int(sim_device)  # Ensure it's an integer
            except ValueError:
                raise ValueError(f"Invalid sim_device: {sim_device}. Must be a numeric GPU index.")
        else:
            raise ValueError(f"Invalid sim_device: {sim_device}. Must be a numeric GPU index.")

        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['MUJOCO_sim_device_ID'] = str(sim_device)  # Convert to string for environment variable
        os.environ['sim_device_ID'] = str(sim_device)
        log.info(f"Set sim_device={sim_device} from cfg.")
    else:
        os.environ['MUJOCO_GL'] = 'osmesa'
        log.info("No EGL device specified in cfg, falling back to osmesa.")

    # For pre-training: download dataset if needed
    if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
        download_url = get_dataset_download_url(cfg)
        download_target = os.path.dirname(cfg.train_dataset_path)
        log.info(f"Downloading dataset from {download_url} to {download_target}")
        gdown.download_folder(url=download_url, output=download_target)

    # For fine-tuning: download normalization if needed
    if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
        download_url = get_normalization_download_url(cfg)
        download_target = cfg.normalization_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading normalization statistics from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # For fine-tuning: download checkpoint if needed
    # ReinFlow Authors: specify base_policy_path=null when you wanna resume from an existing fine-tuning checkpoint.
    if "base_policy_path" in cfg and cfg.base_policy_path and (not os.path.exists(cfg.base_policy_path)):
        download_url = get_checkpoint_download_url(cfg)
        if download_url is None:
            raise ValueError(f"Unknown checkpoint path {cfg.base_policy_path}. Did you specify the correct path to the policy you trained?")
        download_target = cfg.base_policy_path
        dir_name = os.path.dirname(download_target)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log.info(f"Downloading checkpoint from {download_url} to {download_target}")
        gdown.download(url=download_url, output=download_target, fuzzy=True)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench
        # import torch
        # torch.cuda.empty_cache()
    
    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()