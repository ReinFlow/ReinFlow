

import hydra
from hydra.utils import to_absolute_path
import os
import json
from omegaconf import OmegaConf
from utils import read_eval_statistics, plot_3d_eval_statistics, plot_eval_statistics
from util.timer import current_time
from util.dirs import REINFLOW_DIR 



@hydra.main(
    version_base=None,
    config_path=os.path.join(REINFLOW_DIR, "agent/eval/visualize/visualize_cfgs"),  # possibly overwritten by --config-path
)
def main(cfg: OmegaConf):
    # Create a copy of the config to modify
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Select environment based on cfg.env.name
    env_name = cfg.env.name
    if env_name not in cfg.env:
        raise ValueError(f"Environment {env_name} not found in configuration. Choose from {list(cfg.env.keys())}")
    eval_paths = cfg.env[env_name].eval_paths

    # Read evaluation statistics
    eval_stats = [read_eval_statistics(to_absolute_path(eval_path)) for eval_path in eval_paths]

    # Create log directory in the original working directory
    logdir = os.path.join(REINFLOW_DIR, f"visualize/{cfg.name}/{cfg.model.name}/{env_name}/{current_time()}")
    os.makedirs(logdir, exist_ok=True)

    # Save configuration as JSON
    config_json_path = os.path.join(logdir, "config.json")
    with open(config_json_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)
    
    # Call the plotting function to show all 6 subplots
    plot_eval_statistics(
        eval_statistics_list=eval_stats,
        inference_steps=cfg.model.pretrain_step,
        model_name=cfg.model.name,
        env_name=env_name,
        labels=cfg.plot.labels,
        colors=cfg.plot.colors,
        add_denoise_step_line=cfg.plot.add_denoise_step_line,
        log_dir=logdir,
        save_file_name=cfg.plot.save_file_name,
        plot_scale=cfg.plot.plot_scale,
        denoising_steps=cfg.plot.get('denoising_steps') #, [1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    
    # show 3d plots.
    plot_3d_eval_statistics(
        eval_statistics_list=eval_stats,
        inference_steps=cfg.model.pretrain_step,
        model_name=cfg.model.name,
        env_name=env_name,
        labels=cfg.plot.labels,
        colors=cfg.plot.colors,
        log_dir=logdir,
        plot_scale=cfg.plot.plot_scale,
        denoising_steps=cfg.plot.get('denoising_steps'), #, [1, 2, 4, 8, 16, 32, 64, 128, 256]
        legend=True,
    )

if __name__ == "__main__":
    main()
