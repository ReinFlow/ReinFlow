## Implementation Details

---

### ReinFlow Implementation Details

#### Key Hyperparameters

- `min_std` and `max_std`: Minimum and maximum standard deviation of the noise injected at each denoising step.
- `denoising_steps`: Number of denoising steps.
- `ft_denoising_steps`: Number of denoising steps to be fine-tuned, counting backward from the last step. Defaults to the value of `denoising_steps`.

#### Additional Hyperparameters

- `train.clip_intermediate_actions`: Whether to clip intermediate actions during inference. Enable during fine-tuning and evaluation.
- `model.denoised_clip_value`: Maximum absolute value for denoised actions. Default is 1.
- `model.randn_clip_value`: Each sampled denoising action should fall within this many standard deviations from the mean. This prevents extremely large actions. Default is 3.
- `model.clip_ploss_coef`: PPO clipping parameter $\epsilon$. For robotics with complex policies, smaller $\epsilon$ values are preferred. Following DPPO, we use $\epsilon = 0.01$ for state-input tasks and $\epsilon = 0.001$ for visual manipulation. This parameter strongly affects policy gradient stability.
- `model.logprob_min` and `model.logprob_max`: Clipping ranges for log probabilities. If your hyperparameters are set correctly, log probabilities should remain within this range. If the policy collapses, log probabilities may become extremely low (negative). In such cases, consider reducing the noise level.
- `model.noise_scheduler_type`: Type of noise scheduler. For state-input locomotion tasks, it is beneficial to start with higher noise std bounds and then decrease them for better convergence (`learn_decay`). Use `learn` to keep noise bounds constant and let the policy fit the noise automatically. Use `constant` for constant noise.
- `model.use_time_independent_noise`: Set to `False` if noise depends on observations and time; otherwise, set to `True`. This also affects the network architecture.
- `model.noise_hidden_dims`: List of hidden dimensions for the noise injection network.
- `model.explore_net_activation_type`: Activation function for the noise network. Eight nonlinearities are supported, as defined in [model/common/mlp.py](../model/common/mlp.py).
- `model.learn_explore_time_embedding`: Whether to train a time encoder for the noise network from scratch. Default is `False` (sharing the pre-trained policy's time encoder is sufficient).
- `model.time_dim_explore`: Dimension of a separate time encoder, if trained. Default is 0 (no separate encoder).
- `model.critic.out_bias_init`: If your critic is poorly initialized (e.g., outputs negative values despite a 30% policy success rate), set this to a positive float to add an initial bias to the final linear layer of your MLP or ViT. We set this to 4.0 only for the transport-image task.

- `train.use_bc_loss`, `train.bc_loss_type`, `train.bc_loss_coeff`: Whether to use behavior cloning (BC) loss as regularization, the type of BC loss (`None` or `'W2'`), and its intensity (e.g., 1.0). BC regularization is generally unnecessary except for the hopper task.
- `train.ent_coef`: Entropy coefficient $\alpha_{\mathbf{h}}$. Use $\alpha_{\mathbf{h}} = 0.03$ for state-input tasks and 0 for visual manipulation to ensure stability. Larger $\alpha_{\mathbf{h}}$ increases noise std from `min_std` to `max_std`, while zero causes noise to decrease from `max_std` to `min_std`.

---

### DPPO Implementation Details

#### Key Hyperparameters

- `denoising_steps`: Number of denoising steps (should be the same for pre-training and fine-tuning, regardless of the fine-tuning scheme).
- `ft_denoising_steps`: Number of fine-tuned denoising steps.
- `horizon_steps`: Predicted action chunk size (should match `act_steps` for MLP; can differ for UNet, e.g., `horizon_steps=16`, `act_steps=8`).
- `model.gamma_denoising`: Denoising discount factor.
- `model.min_sampling_denoising_std`: Minimum noise when sampling at a denoising step.
- `model.min_logprob_denoising_std`: Minimum standard deviation when evaluating likelihood at a denoising step.
- `model.clip_ploss_coef`: PPO clipping ratio.
- `train.batch_size`: The batch size is relatively large due to PPO updates being in expectation over both environment and denoising steps (new in v0.6).

#### Fine-tuning DDIM with DPPO

To use DDIM fine-tuning, set `denoising_steps=100` in pre-training, and set `model.use_ddim=True`, `model.ddim_steps` to the desired total DDIM steps, and `ft_denoising_steps` to the desired number of fine-tuned DDIM steps.  
For example, in our Furniture-Bench experiments, we use `denoising_steps=100`, `model.ddim_steps=5`, and `ft_denoising_steps=5`.

#### How does DPPO calculate log probabilities for DDIM?**  
DPPO does not fine-tune a deterministic DDIM policy. Their DDIM uses `eta=1`, which closely resembles a DDPM, though the steps and coefficients differ. In contrast, ReinFlow directly processes a flow model with an ODE path.  See [this post](https://github.com/irom-princeton/dppo/issues/48) for details.

#### DPPO Code Implementations

**We provide two DPPO implementations**:
- The official DPPO code: [agent/finetune/dppo/train_ppo_diffusion_agent.py](../agent/finetune/dppo/train_ppo_diffusion_agent.py)
- The ReinFlow authors' version: [agent/finetune/reinflow/train_ppo_diffusion_agent.py](../agent/finetune/reinflow/train_ppo_diffusion_agent.py) and [agent/finetune/reinflow/train_ppo_diffusion_img_agent.py](../agent/finetune/reinflow/train_ppo_diffusion_img_agent.py), which adds verbose logging, training resume, more efficient memory allocation, and a clearer structure. This implementation also leverages inheritance for easier development.

**Our recommendation**:
We recommend using our version for a more flexible and maintainable implementation.  
You are welcome to inherit our base class in [agent/finetune/reinflow/train_agent.py](../agent/finetune/reinflow/train_agent.py) to develop your own RL post-training algorithms.

---

### FQL Implementation Details

#### Key Hyperparameters

- `offline_steps`: Number of iterations for offline fine-tuning. For fair comparison, use the same number of data samples as in the pre-trained checkpoints of the online RL methods.
- `online_steps`: Number of iterations for online fine-tuning.
- `eval_base_model`: For verbose debugging, allows inspection of base model evolution during both offline and online training. Set to `False` if unnecessary.

---

### Offline RL Baselines

We inherit code from [DPPO](https://github.com/irom-princeton/dppo) and include training scripts for offline RL baselines in [../agent/finetune/offlinerl_baselines](../agent/finetune/offlinerl_baselines):

- [Cal-QL](../agent/finetune/offlinerl_baselines/train_calql_agent.py)
- [IBRL](../agent/finetune/offlinerl_baselines/train_ibrl_agent.py)
- [RLPD](../agent/finetune/offlinerl_baselines/train_rlpd_agent.py)
<!-- Add more as needed -->

---

### Offline-to-Online RL Baselines

It is possible to develop additional offline-to-online RL baselines based on the FQL implementation.  
We leave this for future work and welcome community contributions!

---

### Diffusion x RL Baselines

We also inherit diffusion policy RL baselines from [DPPO](https://github.com/irom-princeton/dppo).  
You can find the training scripts in [agent/finetune/diffusion_baselines](../agent/finetune/diffusion_baselines):

- [RWR](agent/finetune/diffusion_baselines/train_rwr_diffusion_agent.py)
- [DAWR](agent/finetune/diffusion_baselines/train_awr_diffusion_agent.py)
- [DIPO](agent/finetune/diffusion_baselines/train_dipo_diffusion_agent.py)
- [DQL](agent/finetune/diffusion_baselines/train_dql_diffusion_agent.py)
- [IDQL](agent/finetune/diffusion_baselines/train_idql_diffusion_agent.py)
- [QSM](agent/finetune/diffusion_baselines/train_qsm_diffusion_agent.py)
