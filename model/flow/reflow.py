
"""
1-Rectified Flow Policy
"""


import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_flow import FlowMLP

log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")

class ReFlow(nn.Module):
    def __init__(
        self,
        network: FlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        sample_t_type: str = 'uniform'
    ):
        """Initialize the ReFlow model with specified parameters.

        Args:
            network: FlowMLP network for velocity prediction.
            device: Device to run the model on (e.g., 'cuda' or 'cpu').
            horizon_steps: Number of steps in the trajectory horizon.
            action_dim: Dimension of the action space.
            act_min: Minimum action value for clipping.
            act_max: Maximum action value for clipping.
            obs_dim: Dimension of the observation space.
            max_denoising_steps: Maximum number of denoising steps for sampling.
            seed: Random seed for reproducibility.
            batch_size: Batch size for training and sampling.
            sample_t_type: Type of time sampling ('uniform', 'logitnormal', 'beta').

        Raises:
            ValueError: If max_denoising_steps is not a positive integer.
        """
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be a positive integer')
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.network = network.to(device)
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.act_range = (act_min, act_max)
        self.obs_dim = obs_dim
        self.max_denoising_steps = int(max_denoising_steps)
        self.sample_t_type = sample_t_type

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0.

        Args:
            x1: Target data tensor of shape (batch_size, horizon_steps, action_dim).
            x0: Initial noise tensor of shape (batch_size, horizon_steps, action_dim).
            t: Time step tensor of shape (batch_size,).

        Returns:
            Tensor: Interpolated trajectory xt of shape (batch_size, horizon_steps, action_dim).
        """
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(self.device)  # ReinFlow Authors revised on 04/23/2025
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1).

        Args:
            batch_size: Number of time samples to generate.
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
            **kwargs: Additional parameters for non-uniform distributions.

        Returns:
            Tensor: Time samples of shape (batch_size,).

        Raises:
            ValueError: If time_sample_type is not supported.
        """
        supported_time_sample_type = ['uniform', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)   # Default beta
            s = kwargs.get("s", 0.999)       # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field.

        Args:
            x1: Real data tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            tuple: Contains (xt, t, obs) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
                - t: Time step tensor of shape (batch_size,).
                - v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).
        """
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    def loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
        """Compute the MSE loss between predicted and target velocities.

        Args:
            xt: Corrupted data tensor of shape (batch_size, horizon_steps, action_dim).
            t: Time step tensor of shape (batch_size,).
            obs: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            v: Target velocity tensor of shape (batch_size, horizon_steps, action_dim).

        Returns:
            Tensor: Mean squared error loss.
        """
        v_hat = self.network(xt, t, obs)
        return F.mse_loss(input=v_hat, target=v)
    
    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True,
        z:torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.

        Args:
            cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
            inference_steps: Number of denoising steps.
            record_intermediate: Whether to return intermediate predictions.
            clip_intermediate_actions: Whether to clip actions to act_range.

        Returns:
            Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        steps = torch.linspace(0, 1, inference_steps, device=self.device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.network(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)