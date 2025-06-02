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


# MIT License

# Copyright (c) 2024 Kevin Frans

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

# The description of ShortCutFlowMLP is translated from Kevin Fran's One Step Diffusion via Short Cut Models 
# and revised by ReinFlow Authors and Collaborators. 
# The ShortCutFlowViT scipt is extended from Diffusion Policy Policy Optimization's implementation. 
# NoisyShortCutFlowMLP and NoisyShortCutFlowViT are composed by ReinFlow Authors.


import logging
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple
from model.flow.mlp_shortcut import ShortCutFlowMLP
log = logging.getLogger(__name__)
Sample = namedtuple("Sample", "trajectories chains")

class ShortCutFlow(nn.Module):
    def __init__(
        self,
        network: ShortCutFlowMLP,
        device: torch.device,
        horizon_steps: int,
        action_dim: int,
        act_min: float,
        act_max: float,
        obs_dim: int,
        max_denoising_steps: int,
        seed: int,
        self_consistency_k=0.25,
        delta:float=1e-5,
        sample_t_type: str = 'uniform'        
    ):
        super().__init__()
        if int(max_denoising_steps) <= 0:
            raise ValueError('max_denoising_steps must be positive integer')
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
        self.max_denoising_steps = max_denoising_steps
        self.self_consistency_k = self_consistency_k
        self.delta = delta
        self.sample_t_type = sample_t_type

    def loss(self, x1: Tensor, cond: dict) -> Tensor:
        """
        Compute the combined loss for flow-matching (d=0) and self-consistency (d>0).

        Args:
            x1: (B, Ta, Da) - Real action trajectories
            cond: dict with 'state' - Observations (B, To, Do)
        
        Returns:
            loss: Scalar tensor - MSE between predicted and target velocities
        """
        
        B = x1.shape[0]
        k_num = int(B * self.self_consistency_k)

        # Self-consistency part. `_sc` means self-consistency. 
        x_sc_1 = x1[:k_num]
        d_base_sc = torch.randint(0, self.max_denoising_steps - 1, (k_num,), device=self.device)
        d_sc = 1.0 / (2 ** d_base_sc.float())
        d_boostrap_sc = d_sc / 2
        dt_sections_sc = 2 ** d_base_sc
        # print(f"d_base_sc={d_base_sc}, dt_sections_sc={dt_sections_sc}")
        
        t_idx_sc = torch.cat([ torch.randint(0, int(dt_sections_sc[i]), (1,), device=self.device) for i in range(k_num)], dim=-1)
        t_sc = t_idx_sc.float() / dt_sections_sc
        x0_sc = torch.randn_like(x_sc_1, device=self.device)
        t_full_sc = t_sc[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
        x_t_sc = (1 - (1 - self.delta) * t_full_sc) * x0_sc + t_full_sc * x_sc_1
        cond_sc = {key:value[:k_num] for key, value in cond.items()}
        with torch.no_grad():
            v_t_sc = self.network.forward(x_t_sc, t_sc, d_boostrap_sc, cond_sc)
            d_boostrap_full = d_boostrap_sc[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
            x_t_sc2 = x_t_sc + v_t_sc * d_boostrap_full
            t2_sc = t_sc + d_boostrap_sc
            v_t_sc2 = self.network.forward(x_t_sc2, t2_sc, d_boostrap_sc, cond_sc)
            v_target_sc = (v_t_sc + v_t_sc2) / 2

        # Flow-matching part
        x_fm_1 = x1[k_num:]
        t_fm = torch.rand((B - k_num,), device=self.device)
        x0_fm = torch.randn_like(x_fm_1, device=self.device)
        t_full_fm = t_fm[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
        x_t_fm = (1 - (1 - self.delta) * t_full_fm) * x0_fm + t_full_fm * x_fm_1
        v_target_fm = x_fm_1 - (1 - self.delta) * x0_fm
        d_fm = torch.zeros((B - k_num,), device=self.device)
        
        # Combine to a whole batch
        x_t = torch.cat([x_t_sc, x_t_fm], dim=0)
        t = torch.cat([t_sc, t_fm], dim=0)
        d = torch.cat([d_sc, d_fm], dim=0)
        v_target = torch.cat([v_target_sc, v_target_fm], dim=0)

        # Predict and compute loss
        v_pred = self.network.forward(x_t, t, d, cond)
        loss   = F.mse_loss(v_pred, v_target)
        return loss 
    
    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True
    ) -> Sample:
        """
        Sample action trajectories using the learned shortcut velocity field. We will use Euler integrator.

        Args:
            cond: dict with 'state' - Observations (B, To, Do)
            inference_steps: Number of denoising steps
            record_intermediate: Whether to record intermediate trajectories
            clip_intermediate_actions: Whether to clip actions to act_range

        Returns:
            Sample: Named tuple with 'trajectories' and optional 'chains'
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = torch.randn((B,) + self.data_shape, device=self.device)
        dt = 1.0 / inference_steps
        t = torch.linspace(0, 1 - dt, inference_steps, device=self.device)
        d = torch.full((B,), dt, device=self.device)

        for i in range(inference_steps):
            t_i = t[i] * torch.ones(B, device=self.device)
            vt = self.network.forward(x_hat, t_i, d, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)
    
    
    
