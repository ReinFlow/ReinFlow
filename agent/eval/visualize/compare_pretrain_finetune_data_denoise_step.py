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


import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import json
from util.dirs import REINFLOW_DIR 
from util.process import sort_handles_by_labels
import os
import numpy as np
from utils import read_eval_statistics, plot_3d_eval_statistics_groups, plot_3d_eval_statistics_groups_same_color
from util.timer import current_time
import logging 
log = logging.getLogger(__name__)

"""
This script compares the performance of pretrained and finetuned models in terms of denoising steps, number of 
pretraining episodes, and evaluation metrics (success rate, average episode reward, and trajectory length). It loads 
evaluation statistics from .npz files specified in a Hydra configuration file, computes the mean and standard deviation 
of finetuned statistics across three random seeds for each episode number, and generates 3D surface plots to visualize 
the comparison between pretrained and finetuned models. The finetuned surface uses shading (via alpha transparency) to 
indicate standard deviation across seeds, with purple-toned color maps (darker for finetuned, lighter for pretrained). 
Plots are saved as PNG and PDF files in a timestamped log directory, and the configuration is saved as JSON for reproducibility.
"""

@hydra.main(
    version_base=None,
    config_path=os.path.join(REINFLOW_DIR, "agent/eval/visualize/visualize_cfgs"),
    config_name="hopper_datascale_finetune.yaml" #"square_datascale_finetune" # "hopper_datascale_finetune.yaml"
)
def main(cfg: OmegaConf):
    """
    Main function to load configuration, process evaluation statistics, and generate 3D surface plots comparing
    pretrained and finetuned models. It computes the mean and standard deviation of finetuned statistics across three 
    random seeds (0, 42, 3407) for each episode number and visualizes the results using purple-toned surfaces (darker 
    purple for finetuned with std-based shading, lighter purple for pretrained).

    Args:
        cfg (OmegaConf): Hydra configuration object containing model, environment, and plot settings, including
                         paths to evaluation statistics files and plotting parameters.
    """
    # Create a copy of the config to modify
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Select environment based on cfg.env.name
    env_name = cfg.env.name
    if env_name not in cfg.env:
        raise ValueError(f"Environment {env_name} not found in configuration. Choose from {list(cfg.env.keys())}")
    
    # Extract pretrained and finetuned eval_paths
    pretrained_paths = cfg.env[env_name].eval_paths.pretrained
    finetuned_paths = cfg.env[env_name].eval_paths.finetuned
    
    # Read pretrained evaluation statistics with error handling
    pretrained_stats = []
    for path in pretrained_paths:
        try:
            stats = read_eval_statistics(to_absolute_path(path))
            pretrained_stats.append(stats)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained eval file not found: {path}")
        except PermissionError:
            raise PermissionError(f"Permission denied for pretrained eval file: {path}")
        except Exception as e:
            raise Exception(f"Error loading pretrained eval file {path}: {str(e)}")
    
    # Group finetuned paths by episode number
    episode_numbers = cfg.plot.labels
    finetuned_paths_grouped = {ep: [] for ep in episode_numbers}
    for path in finetuned_paths:
        for ep in episode_numbers:
            if f"/{ep}/" in path:
                finetuned_paths_grouped[ep].append(path)
                break
    print(f"finetuned_paths_grouped={finetuned_paths_grouped}")
    
    # Read and compute mean and std for finetuned evaluation statistics across three seeds
    finetuned_stats = []
    for ep in episode_numbers:
        paths = finetuned_paths_grouped[ep]
        if len(paths) != 3:
            # raise ValueError(f"Expected 3 seeds for episode {ep}, got {len(paths)}")
            log.info(f"Warning: Expected 3 seeds for episode {ep}, got {len(paths)}")
        stats_list = []
        for path in paths:
            try:
                stats = read_eval_statistics(to_absolute_path(path))
                stats_list.append(stats)
            except FileNotFoundError:
                raise FileNotFoundError(f"Finetuned eval file not found: {path}")
            except PermissionError:
                raise PermissionError(f"Permission denied for finetuned eval file: {path}")
            except Exception as e:
                raise Exception(f"Error loading finetuned eval file {path}: {str(e)}")
        
        # Compute mean and std across the three seeds for each metric
        # stats_list contains tuples: (denoising_steps, metric1, metric2, ...)
        # We assume denoising_steps are identical across seeds
        denoising_steps = stats_list[0][0]
        mean_stats = [denoising_steps]
        std_stats = [np.zeros_like(denoising_steps)]  # Std for denoising_steps is 0
        for i in range(1, len(stats_list[0])):
            # Compute mean and std for each metric across the three seeds
            metric_values = np.array([stats[i] for stats in stats_list])
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            mean_stats.append(mean_values)
            std_stats.append(std_values)
        # Store mean and std as a single tuple
        finetuned_stats.append((tuple(mean_stats), tuple(std_stats)))
    
    # Prepare data for plotting
    eval_statistics_groups = [finetuned_stats, pretrained_stats]
    group_labels = ['Fine-tuned', 'Pre-trained']
    
    # Create log directory
    logdir = os.path.join(REINFLOW_DIR, f"visualize/{cfg.name}/{cfg.model.name}/{env_name}") #{current_time()}
    os.makedirs(logdir, exist_ok=True)

    # Save configuration as JSON
    config_json_path = os.path.join(logdir, "config.json")
    with open(config_json_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
    
    # Generate 3D plots with purple-toned color maps
    plot_3d_eval_statistics_groups_same_color(
        eval_statistics_groups=eval_statistics_groups,
        group_labels=group_labels,
        inference_steps=cfg.model.pretrain_step,
        model_name=cfg.model.name,
        env_name=env_name,
        labels=cfg.plot.labels,
        colors=cfg.plot.colors,
        log_dir=logdir,
        save_file_name=cfg.plot.save_file_name,
        plot_scale=cfg.plot.plot_scale,
        denoising_steps=cfg.plot.get('denoising_steps'),
        legend=True,
        plot_additional_metrics=True,
        # finetuned_cmap='Purples',  # Darker purple for finetuned
        # pretrained_cmap='PuRd'     # Lighter purple (light velvet) for pretrained
    )

if __name__ == "__main__":
    main()