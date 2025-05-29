"""
MLP models for flow matching with learnable stochastic interpolate noise.
"""
import torch
import torch.nn as nn
import logging
import numpy as np
from copy import deepcopy
from typing import Tuple
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.vit import VitEncoder
log = logging.getLogger(__name__)
import einops
from typing import List

class FlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim=action_dim
        self.cond_dim=cond_dim
        self.mlp_dims=mlp_dims
        self.activation_type=activation_type
        self.out_activation_type=out_activation_type
        self.use_layernorm=use_layernorm
        self.residual_style=residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        model = ResidualMLP if residual_style else MLP
        
        # obs encoder
        if cond_mlp_dims:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim
        input_dim = time_dim + action_dim * horizon_steps + self.cond_enc_dim
        
        # velocity head
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond,
        output_embedding=False,
        **kwargs,
    ):
        """
        **Args**:
            action: (B, Ta, Da)
            time: (B,) or int, diffusion step
            cond: dict with key state/rgb; more recent obs at the end
                    state: (B, To, Do)
        **Outpus**:
            velocity. 
            vel: (B, Ta, Da) when output_embedding==False 
            vel,time_emb, cond_emb: when output_embedding==False
        """
        B, Ta, Da = action.shape

        # flatten action chunk
        action = action.view(B, -1)

        # flatten obs history
        state = cond["state"].view(B, -1)

        # obs encoder
        cond_emb = self.cond_mlp(state) if hasattr(self, "cond_mlp") else state
        
        # time encoder
        if isinstance(time, int) or isinstance(time, float):
            time=torch.ones((B,1), device=action.device)* time
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        
        # velocity head
        vel_feature = torch.cat([action, time_emb, cond_emb], dim=-1)
        vel = self.mlp_mean(vel_feature)
        
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_emb
        return vel.view(B, Ta, Da)

    def sample_action(self,cond:dict,inference_steps:int,clip_intermediate_actions:bool,act_range:List[float], z:Tensor=None,save_chains:bool=False):
        """
        simply return action via integration (Euler's method). the initial noise could be specified. 
        when `save_chains` is True, also return the denoising trajectory.
        """
        B = cond['state'].shape[0]
        device=cond['state'].device

        x_hat:Tensor=z if z is not None else torch.randn(B, self.horizon_steps, self.action_dim, device=device)
        if save_chains:
            x_chain=torch.zeros((B, inference_steps+1, self.horizon_steps, self.action_dim), device=device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=device)
        steps = torch.linspace(0, 1, inference_steps, device=device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.forward(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*act_range)
            if save_chains:
                x_chain[:, i+1] = x_hat
        if save_chains:
            return x_hat, x_chain
        return x_hat
        
    # not stable yet. needs some checking. 
    def compute_logprobability(self, action: torch.Tensor, cond: dict, num_inference_steps: int) -> torch.Tensor:
        """
        Compute the log probability log p_1(action | cond) using the flow model.

        This method integrates the flow ODE backwards from t=1 to t=0 using Euler's method
        and estimates the divergence with the Hutchinson estimator to compute the likelihood.

        Args:
            action (Tensor): Shape (B, T_a, D_a), the action to evaluate.
            cond (dict): Conditioning info, e.g., {'state': Tensor of shape (B, To, Do)}.
            num_inference_steps (int): Number of discretization steps for ODE integration.

        Returns:
            Tensor: Log probabilities, shape (B,).
        
        Reference: https://github.com/facebookresearch/flow_matching
        """
        # Extract dimensions and device
        B, T_a, D_a = action.shape
        device = action.device
        dt = 1.0 / num_inference_steps

        # Initialize state and log-likelihood term
        x = action.clone().requires_grad_(True)  # Shape: (B, T_a, D_a)
        g = torch.zeros(B, device=device)       # Shape: (B,)

        # Sample random vector z for Hutchinson estimator (fixed across steps)
        eps = torch.randn_like(x)  # Shape: (B, T_a, D_a)
        
        # Backward integration from t=1 to t=0
        for k in range(num_inference_steps - 1, -1, -1):
            t = (k + 1) / num_inference_steps # Scalar time from 1 to dt
            with torch.set_grad_enabled(True):
                # Compute the vector field u_t(x)
                u = self.forward(x, t, cond)  # Shape: (B, T_a, D_a)

                # Compute Hutchinson estimator for divergence
                # grad_u_z = (\partial_x u)^T z
                grad_u_z = torch.autograd.grad(
                    outputs=u,
                    inputs=x,
                    grad_outputs=eps,
                    create_graph=True
                )[0]  # Shape: (B, T_a, D_a)
                
                # Estimate tr(\partial_x u) ≈ z^T (\partial_x u)^T z
                div_estimator = (grad_u_z * eps).sum(dim=(1, 2))  # Shape: (B,)
            
            # Euler step backwards
            x = x + (-dt) * u.detach()        # Update state
            g = g + dt * div_estimator.detach()  # Accumulate log-likelihood change

        # Compute log probability of the source distribution (standard normal)
        log_p0 = -0.5 * torch.sum(x**2, dim=(1, 2)) - (T_a * D_a / 2) * np.log(2 * np.pi)
        
        # Final log probability: log p_1(x) = log p_0(f(0)) - \int_0^1 div(u_t) dt
        log_p1 = log_p0 - g
        return log_p1
    
    
class ExploreNoiseNet(nn.Module):
    '''
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings. 
    \sigma(s,t) or \sigma(s)
    '''
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 logprob_denoising_std_range:list, #[min_std, max_std]
                 device,
                 hidden_dims=[16], #[8]  [32],
                 activation_type='Tanh'
                 ):
        super().__init__()
        self.device = device
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims +[out_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
        ).to(self.device)
        
        self.set_noise_range(logprob_denoising_std_range)
    
    def set_noise_range(self, logprob_denoising_std_range:list):
        self.logprob_denoising_std_range=logprob_denoising_std_range
        min_logprob_denoising_std = self.logprob_denoising_std_range[0]
        max_logprob_denoising_std = self.logprob_denoising_std_range[1]
        self.logvar_min = torch.nn.Parameter(torch.log(torch.tensor(min_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
        self.logvar_max = torch.nn.Parameter(torch.log(torch.tensor(max_logprob_denoising_std**2, dtype=torch.float32, device=self.device)), requires_grad=False)
    
    def forward(self, noise_feature:torch.Tensor):
        '''
        '''
        noise_logvar    = self.mlp_logvar(noise_feature)
        noise_std       = self.process_noise(noise_logvar)
        return noise_std

    def process_noise(self, noise_logvar):
        '''
        input:
            torch.Tensor([B, Ta , Da])   log \sigma^2 
        output:
            torch.Tensor([B, 1, Ta * Da]), sigma, floating point values, bounded in [min_logprob_denoising_std, max_logprob_denoising_std]
        '''
        noise_logvar = noise_logvar
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1)/2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


class NoisyFlowMLP(nn.Module):
    def __init__(
        self,
        policy:FlowMLP,
        denoising_steps,
        learn_explore_noise_from,
        inital_noise_scheduler_type,
        min_logprob_denoising_std,
        max_logprob_denoising_std,
        learn_explore_time_embedding,
        time_dim_explore,
        use_time_independent_noise,
        device,
        noise_hidden_dims=None,
        activation_type='Tanh'
    ):  
        super().__init__()
        self.device=device
        self.policy:FlowMLP = policy.to(self.device)
        """
        input:  [batchsize, time_dim + cond_enc_dim]
        output: positive tensor of shape [batchsize, self.denoising_steps, self.horizon_steps x self.act_dim]
        """
        
        self.denoising_steps: int = denoising_steps
        self.learn_explore_noise_from: int = learn_explore_noise_from
        self.initial_noise_scheduler_type: str = inital_noise_scheduler_type
        if min_logprob_denoising_std > max_logprob_denoising_std:
            raise ValueError(f"min_logprob_denoising_std must not exceed max_logprob_denoising_std, but received min_logprob_denoising_std={min_logprob_denoising_std} > max_logprob_denoising_std={max_logprob_denoising_std}. Revise your configuration file!")
        self.min_logprob_denoising_std: float = min_logprob_denoising_std
        self.max_logprob_denoising_std: float = max_logprob_denoising_std    
        self.learn_explore_time_embedding: bool  = learn_explore_time_embedding
        self.set_logprob_noise_levels()
        
        self.noise_hidden_dims=noise_hidden_dims
        self.use_time_independent_noise = use_time_independent_noise
        self.time_dim_explore =time_dim_explore
        self.noise_activation_type=activation_type
        self.init_exploration_noise_net()
        
    def init_exploration_noise_net(self):
        if self.use_time_independent_noise:
            noise_input_dim = self.policy.cond_enc_dim
            if not self.noise_hidden_dims:
                self.noise_hidden_dims = [16]
        else:
            if self.learn_explore_time_embedding:
                noise_input_dim = self.time_dim_explore + self.policy.cond_enc_dim
                self.time_embedding_explore = nn.Embedding(num_embeddings=self.denoising_steps, 
                                                       embedding_dim = self.time_dim_explore, 
                                                       device=self.device)
            else:
                noise_input_dim = self.policy.time_dim + self.policy.cond_enc_dim
                if not self.noise_hidden_dims:
                    self.noise_hidden_dims = [int(np.sqrt(noise_input_dim**2 + self.policy.act_dim_total**2))]
        
        self.explore_noise_net=ExploreNoiseNet(in_dim=noise_input_dim, 
                                                out_dim=self.policy.act_dim_total,
                                                logprob_denoising_std_range=[self.min_logprob_denoising_std, self.max_logprob_denoising_std], 
                                                device=self.device,
                                                hidden_dims=self.noise_hidden_dims,
                                                activation_type=self.noise_activation_type)
    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        inputs:
            x: (B, Ta, Da)
            time: (B,) floating point in [0,1) flow matching time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching inference step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        vel, time_emb, cond_emb = self.policy.forward(action, time, cond, output_embedding=True)
        
        # noise head (for exploration). allow gradient flow.
        if self.initial_noise_scheduler_type=='const' or step < self.learn_explore_noise_from:
            noise_std       = self.logprob_noise_levels[:, step].repeat(B,1)
        else:
            if self.use_time_independent_noise:
                noise_feature    = cond_emb
            else:
                if self.learn_explore_time_embedding:
                    step_ts = torch.tensor(step, device = self.device).repeat(B)
                    time_emb_explore = self.time_embedding_explore(step_ts)
                    noise_feature    = torch.cat([time_emb_explore, cond_emb], dim=-1)
                else:
                    noise_feature    = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
            
            if verbose:
                log.info(f"step={step}, learnable noise = {noise_std.mean()}")
        
        if verbose:
            log.info(f"step={step}, set to learn from {self.learn_explore_noise_from}, will learn exploration noise ? {step >= self.learn_explore_noise_from}, noise_std={noise_std.mean()}require_grad={noise_std.requires_grad}")
        
        return vel, noise_std if learn_exploration_noise else noise_std.detach()

    @torch.no_grad()
    def stochastic_interpolate(self,t):
        valid_noise_schedulers=['vp', 'lin', 'const', 'const_schedule_itr', 'learn_decay']
        if self.initial_noise_scheduler_type == 'vp':
            a = 0.2 #2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.initial_noise_scheduler_type == 'lin':
            k=0.1
            b=0.0
            std = k*t+b
        elif self.initial_noise_scheduler_type == 'const' or 'const_schedule_itr':
            std = torch.ones_like(t) * self.min_logprob_denoising_std
        else:
            raise ValueError(f"Invalid noise scheduler type {self.initial_noise_scheduler_type}, must be in the following: {valid_noise_schedulers}")
        return std
    
    @torch.no_grad()
    def set_logprob_noise_levels(self, force_level=None, verbose=False):
        '''
        create noise std for logrporbability calcualion. 
        generate a tensor `self.logprob_noise_levels` of shape `[1, self.denoising_steps,  self.policy.horizion_steps x self.policy.act_dim]`
        '''
        self.logprob_noise_levels = torch.zeros(self.denoising_steps, device=self.device, requires_grad=False)
        
        steps = torch.linspace(0, 1, self.denoising_steps, device=self.device)
        for i, t in enumerate(steps):
            if force_level:
                self.logprob_noise_levels[i] = torch.tensor(force_level, device=self.device)
            else:
                self.logprob_noise_levels[i] = self.stochastic_interpolate(t)
        
        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std, max=self.max_logprob_denoising_std)
        
        self.logprob_noise_levels = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.policy.horizon_steps *  self.policy.action_dim)
        
        if verbose:
            log.info(f"Set logprob noise levels. self.logprob_noise_levels={self.logprob_noise_levels}")

class VisionFlowMLP(nn.Module):
    """With ViT backbone"""
    def __init__(
        self,
        backbone: VitEncoder,
        action_dim,
        horizon_steps,
        cond_dim,                       # proprioception only
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=0,
        visual_feature_dim=128,         # visual feature dim
        dropout=0,
        num_img=1,                      # currently only supports 1 or 2
        augment=False,
    ):
        super().__init__()
        
        # action chunk
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps
        
        # historical proprioception and visual inputs
        self.prop_dim = cond_dim    
        self.img_cond_steps = img_cond_steps
        
        # time
        self.time_dim = time_dim
        
        self.backbone = backbone
        self.mlp_dims = mlp_dims
        self.activation_type = activation_type
        self.out_activation_type = out_activation_type
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style
        self.spatial_emb = spatial_emb
        
        self.dropout = dropout
        self.num_img = num_img
        self.augment = augment
        
        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img == 2:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                raise NotImplementedError(f"num_img={num_img} Currently we only support 1 or 2 image inputs")
            visual_feature_dim = spatial_emb * num_img
        else: # spatial embedding not specified, use default value 128
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        self.cond_enc_dim = visual_feature_dim + self.prop_dim     # rgb and  proprioception      
        
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Flow
        input_dim = (
            time_dim + \
                action_dim * horizon_steps + \
                        self.cond_enc_dim
        )
        
        # output action chunk
        output_dim = action_dim * horizon_steps
        
        # velocity head
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        cond: dict,
        output_embedding=False,
        **kwargs,
    ):
        """
        inputs:
            action: (B, Ta, Da) action chunk
            time: (B,) or float within [0,1), flow time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        outputs:

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = action.shape
        _, T_rgb, C, H, W = cond["rgb"].shape
        # flatten chunk
        action = action.view(B, -1)

        # flatten history (proprioception, here we use the raw input without encoding)
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]
        # concatenate images in cond by channels
        if self.num_img >1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img==1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"self.num_img={self.num_img} <1. ")
        # convert rgb to float32 for augmentation
        rgb = rgb.float()
        
        # visual and proprioceptive embeddings: get vit output - pass in two images separately
        if self.num_img ==2:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone.forward(rgb1)
            feat1 = self.compress1.forward(feat1, state)
            
            feat2 = self.backbone.forward(rgb2)
            feat2 = self.compress2.forward(feat2, state)
            
            feat = torch.cat([feat1, feat2], dim=-1)
        elif self.num_img ==1:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone.forward(rgb)
            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress.forward(feat, state)
            else:
                feat = feat.flatten(1, -1)
                feat = self.compress(feat)
        else:
            raise NotImplementedError(f"num_img={self.num_img} Currently we only support 1 or 2 image inputs")
        cond_encoded = torch.cat([feat, state], dim=-1)   # visual and proprioception inputs. 

        # time embedding
        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        
        # all embeddings: time, visual-proprioceptive
        emb = torch.cat([action, time_emb, cond_encoded], dim=-1)

        # velocity head
        vel = self.mlp_mean(emb)
        if output_embedding:
            return vel.view(B, Ta, Da), time_emb, cond_encoded
        return vel.view(B, Ta, Da)


class NoisyVisionFlowMLP(NoisyFlowMLP):
    def __init__(
            self,
            policy:VisionFlowMLP,
            denoising_steps,
            learn_explore_noise_from,
            inital_noise_scheduler_type,
            min_logprob_denoising_std,
            max_logprob_denoising_std,
            learn_explore_time_embedding,
            time_dim_explore,
            use_time_independent_noise,
            device,
            noise_hidden_dims=None,
            activation_type='Tanh'
    ):
        super().__init__(
            policy,
            denoising_steps,
            learn_explore_noise_from,
            inital_noise_scheduler_type,
            min_logprob_denoising_std,
            max_logprob_denoising_std,
            learn_explore_time_embedding,
            time_dim_explore,
            use_time_independent_noise,
            device,
            noise_hidden_dims,
            activation_type
        )
    
    def forward(
        self,
        action,
        time,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        inputs:
            x: (B, Ta, Da)
            time: (B,) floating point in [0,1) flow matching time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching inference step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        
        self.policy: VisionFlowMLP
        vel, time_emb, cond_emb = self.policy.forward(action, time, cond, output_embedding=True)
        
        # noise head (for exploration). allow gradient flow.
        if self.initial_noise_scheduler_type=='const' or step < self.learn_explore_noise_from:
            noise_std       = self.logprob_noise_levels[:, step].repeat(B,1)
        else:
            if self.use_time_independent_noise:
                noise_feature    = cond_emb
            else:
                if self.learn_explore_time_embedding:
                    step_ts = torch.tensor(step, device = self.device).repeat(B)
                    time_emb_explore = self.time_embedding_explore(step_ts)
                    noise_feature    = torch.cat([time_emb_explore, cond_emb], dim=-1)
                else:
                    noise_feature    = torch.cat([time_emb.detach(), cond_emb], dim=-1)
            # predict noise
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
        
        return vel, noise_std if learn_exploration_noise else noise_std.detach()
    
