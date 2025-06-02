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

import torch
import torch.nn as nn
import numpy as np
import logging
from copy import deepcopy
from torch import Tensor
from model.common.mlp import MLP, ResidualMLP
from model.diffusion.modules import SinusoidalPosEmb
from model.common.modules import SpatialEmb, RandomShiftsAug
from model.common.vit import VitEncoder
log = logging.getLogger(__name__)
import einops
from typing import Tuple
from util.process import is_multiple_of_one_over_k

class ShortCutFlowMLP(nn.Module):
    def __init__(
        self,
        horizon_steps,
        action_dim,
        cond_dim,
        td_emb_dim=16,  # Embedding dimension for time and step
        mlp_dims=[256, 256], # hidden layers of the velocity head.
        cond_mlp_dims=None, # the hidden dimensions and output dimension of condition embedder. 
        activation_type="Mish", # different from reflow (SiLU is better for dense nets)
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        embed_combination_type='add' #, multiply or concate.   cond_embed + td_embed or cond_embed x td_embed or [cond_embed, td_embed]
    ):
        super().__init__()
        self.td_emb_dim = td_emb_dim # for both time and step
        self.act_dim_total = action_dim * horizon_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_dim=cond_dim
        self.activation_type=activation_type
        self.out_activation_type=out_activation_type
        self.time_embed_activation=nn.Mish() # nn.SiLU() maybe better but for fair comparison with reflow and diffusion we use Mish. 
        self.use_layernorm=use_layernorm
        self.residual_style=residual_style
        candidate_embed_combination_types=['add', 'multiply', 'concate']
        if embed_combination_type not in candidate_embed_combination_types:
            raise ValueError(f"embed_combination_type must be one of {candidate_embed_combination_types} but received {embed_combination_type}!")
        self.embed_combination_type=embed_combination_type
        
        # time t and step share an input embedding
        self.map_noise = SinusoidalPosEmb(td_emb_dim)
        # MLP to process concatenated t and step embeddings
        self.t_emb = nn.Sequential(
            nn.Linear(2 * td_emb_dim, td_emb_dim),
            self.time_embed_activation,
            nn.Linear(td_emb_dim, td_emb_dim)
        )
        
        # Condition embedding
        if cond_mlp_dims:
            self.cond_emb = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.cond_enc_dim = cond_mlp_dims[-1]
        else:
            self.cond_enc_dim = cond_dim
        if embed_combination_type in ['add', 'multiply'] and td_emb_dim !=self.cond_enc_dim: 
            raise ValueError(f"To add or multiply td_embed with cond_embed you must make td_emb_dim={td_emb_dim} == self.cond_enc_dim={self.cond_enc_dim}")
        
        
        # velocity head
        model = ResidualMLP if residual_style else MLP
        if self.embed_combination_type =='concate':
            input_dim = action_dim * horizon_steps + self.cond_enc_dim + td_emb_dim #(s,a, t-dt)
        elif self.embed_combination_type =='add' or 'multiply':
            input_dim = action_dim * horizon_steps + td_emb_dim
        else:
            raise ValueError(f"Unsupported embed_combination_type={self.embed_combination_type}")
        self.vel_head = model(
            [input_dim] + mlp_dims + [self.act_dim_total],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )

    def forward(
        self,
        action: Tensor,
        time: Tensor,
        dt: Tensor,
        cond: dict,
        output_embedding=False
    ):
        """
        Inputs:
            action: (B, Ta, Da) - Current action trajectory
            time: (B,) - Current noise level t
            cond: (B, Do) - Condition (e.g., flattened state)
            dt: (B,) - Step size

        Outputs:
            velocity: (B, Ta, Da) - Predicted velocity
        """
        B, Ta, Da = action.shape

        # Flatten action
        action_flat = action.view(B, -1)
        
        # Embed time t and dt separately, then concatenate and feed to the same MLP to squeeze the dimension back to emb_dim
        t_emb = self.map_noise(time.view(B, 1)).view(B, self.td_emb_dim)
        dt_emb = self.map_noise(dt.view(B, 1)).view(B, self.td_emb_dim)
        td_emb = self.t_emb(torch.cat([t_emb, dt_emb], dim=1))

        # Embed condition and add to time-step embedding
        state = cond["state"].view(B, -1)
        cond_emb = self.cond_emb(state) if hasattr(self, "cond_emb") else state
            
        if self.embed_combination_type=='add':# we use add to reduce dimension
            emb = td_emb + cond_emb
        elif self.embed_combination_type=='multiply':# we use add to reduce dimension while preserving nonlinearity
            emb = td_emb * cond_emb
        elif self.embed_combination_type =='concate': # separate the influences of td_embd and cond_emb
            emb=torch.cat([td_emb, cond_emb], dim=-1)
        # Predict velocity
        vel_flat = self.vel_head(torch.cat([action_flat, emb], dim=-1))
        if output_embedding:
            return vel_flat.view(B, Ta, Da), td_emb, cond_emb
        return vel_flat.view(B, Ta, Da)

class ShortCutFlowViT(nn.Module):
    """With ViT backbone and Transformer-based shortcut flow
    
    
    **Architecture**:
    
    camera pixels -> aug-> backbone-> visual_feature - | 
                                                      cat->cond_embed->cond_embedding->|
    proprioception-> prop embedder -> prop_embedding - |                               |
                                                                                       + or * --> cond_td_embedding-|
            t     ->               -> t_embedding ->                                   |                            |
                        map_noise                     td_embed     -->   td_embedding->|                            |
           step   ->               -> dt_embedding->                                                           cat --> vel_head --> vel
                                                                                                                | 
          action  -->   (omitted)                                                         -->  act_embedding   -|
                        (projection + positional embedding)
    """
    def __init__(
        self,
        backbone:VitEncoder,    # VitEncoder instance
        action_dim,
        horizon_steps,
        prop_dim,               # proprioception dimension
        img_cond_steps=1,
        td_emb_dim=16,  # Embedding dimension for time and step
        # d_model=384,  # omitted
        # n_heads=6,
        # depth=12,
        mlp_dims=[256,256], # the hidden dimensions of output velocity head
        cond_mlp_dims=None, # the hidden dimensions and output dimension of condition embedder. 
        activation_type="Mish", # instead of SiLU() maybe better for deep nets.
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        dropout=0.0,
        visual_feature_dim=128, # overload spatial embed when specified.
        num_img=1,
        augment=False,
        spatial_emb=0,
        embed_combination_type='add'  # 'add', 'multiply', or 'concate'
    ):
        super().__init__()
        
        # Action chunk
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.act_dim_total = action_dim * horizon_steps
        
        # Historical proprioception and visual inputs
        self.prop_dim = prop_dim
        self.img_cond_steps = img_cond_steps
        
        # How to combine timestep and condition embeddings.
        candidate_embed_combination_types = ['add', 'multiply', 'concate']
        if embed_combination_type not in candidate_embed_combination_types:
            raise ValueError(f"embed_combination_type must be one of {candidate_embed_combination_types}, got {embed_combination_type}")
        self.embed_combination_type = embed_combination_type
        
        # Transformer dimension (omitted)
        # self.d_model = d_model
        
        # Action embeddings: projection and action chunk positional embedding (omitted)
        # self.x_proj = nn.Linear(action_dim, d_model)
        # self.pos_emb = PositionalEmbedding(d_model)
        
        # Time-step embeddings
        self.td_emb_dim = td_emb_dim
        self.map_noise = SinusoidalPosEmb(td_emb_dim)   # Shared embedding for time t and step size d
        self.time_embed_activation=nn.Mish()            # nn.SiLU() maybe better but for fair comparison with reflow and diffusion we use Mish. 
        self.td_emb = nn.Sequential(
                nn.Linear(2 * td_emb_dim, td_emb_dim),
                self.time_embed_activation,
                nn.Linear(td_emb_dim, td_emb_dim)
            )
        
        # Condition embedding
        if cond_mlp_dims:# add transform to the state 
            self.prop_emb = MLP(
                [prop_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            self.prop_embed_dim = cond_mlp_dims[-1]
        else: # just us the state itself, without transforms. 
            self.prop_embed_dim = prop_dim
        
        # Visual backbone and augmentation
        self.backbone = backbone
        self.num_img = num_img
        self.augment = augment
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        # Visual feature compression
        if spatial_emb > 0:
            assert spatial_emb > 1, "spatial_emb must be > 1"
            if num_img == 2:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=prop_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            elif num_img == 1:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=prop_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            else:
                raise NotImplementedError(f"num_img={num_img} not supported (only 1 or 2)")
            self.visual_feature_dim = spatial_emb * num_img
        else:
            self.visual_feature_dim = visual_feature_dim
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
        self.visuomotor_feature_dim = self.visual_feature_dim + self.prop_embed_dim
        
        if embed_combination_type in ['add', 'multiply']: 
            # compress visuomotor information to the same size of time embedding.
            self.cond_embed=nn.Sequential(
                nn.Linear(self.visuomotor_feature_dim, td_emb_dim*2),
                nn.ReLU(),
                nn.Linear(td_emb_dim*2, td_emb_dim),
            )
            self.cond_enc_dim=td_emb_dim
        else:
            self.cond_enc_dim=self.visuomotor_feature_dim
        
        # Transformer middle blocks (omitted)
        # self.transformer_blocks = nn.ModuleList([
        #     ShortcutDiTBlock(d_model, n_heads, dropout) for _ in range(depth)
        # ])
        

        # velocity head
        vel_head_model = ResidualMLP if residual_style else MLP
        if self.embed_combination_type =='concate':
            input_dim = action_dim * horizon_steps + self.cond_enc_dim + td_emb_dim     #(s, a, t-dt)
        elif self.embed_combination_type =='add' or 'multiply':
            input_dim = action_dim * horizon_steps + self.cond_enc_dim                  
        else:
            raise ValueError(f"Unsupported embed_combination_type={self.embed_combination_type}")
        output_dim = action_dim * horizon_steps
        self.vel_head = vel_head_model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        action,
        time,
        d,
        cond,
        output_embedding=False,
    ):
        """
        Inputs:
            action: (B, Ta, Da) - Action trajectories
            time: (B,) or float - Flow time
            d: (B,) or float - Step size
            cond: dict with keys 'state' and 'rgb'
                state: (B, To, Do) - Proprioceptive states
                rgb: (B, To, C, H, W) - RGB images
            output_embedding: whether also return td_embedding and condition embedding
        Outputs:
            velocity: (B, Ta, Da) - Predicted velocities
        """
        B, Ta, Da = action.shape
        _, T_rgb, C, H, W = cond["rgb"].shape
        
        # flatten chunk
        action_embed = action.view(B, -1)
        # (action transform omitted)
        # Project action chunk and add positional embeddings
        # x = self.x_proj(action)  # (B, Ta, d_model)
        # pos_emb = self.pos_emb(torch.arange(Ta, device=device))
        # x = x + pos_emb[None, :]
        
        # Embed time t and dt separately, then concatenate and feed to the same MLP to squeeze the dimension back to `td_emb_dim`
        t_emb = self.map_noise(time.view(B, 1)).view(B, self.td_emb_dim)
        d_emb = self.map_noise(d.view(B, 1)).view(B, self.td_emb_dim)
        td_emb = self.td_emb(torch.cat([t_emb, d_emb], dim=1))
        
        # Embed proprioceptive states
        state = cond["state"].view(B, -1)
        prop_emb = self.prop_emb(state) if hasattr(self, "prop_emb") else state
        
        # Process visual inputs (augmentation + compression)
        rgb = cond["rgb"][:, -self.img_cond_steps:]
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        elif self.num_img == 1:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")
        else:
            raise ValueError(f"self.num_img={self.num_img} < 1")
        rgb = rgb.float()
        if self.num_img == 2:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            visual_feat1 = self.backbone.forward(rgb1)
            visual_feat1 = self.compress1.forward(visual_feat1, cond["state"].view(B, -1)) if hasattr(self, 'compress1') else self.compress(visual_feat1.flatten(1, -1))
            visual_feat2 = self.backbone.forward(rgb2)
            visual_feat2 = self.compress2.forward(visual_feat2, cond["state"].view(B, -1)) if hasattr(self, 'compress2') else self.compress(visual_feat2.flatten(1, -1))
            visual_feat = torch.cat([visual_feat1, visual_feat2], dim=-1)
        elif self.num_img == 1:
            if self.augment:
                rgb = self.aug(rgb)
            visual_feat = self.backbone.forward(rgb)
            if isinstance(self.compress, SpatialEmb):
                visual_feat = self.compress.forward(visual_feat, cond["state"].view(B, -1))
            else:
                visual_feat = self.compress(visual_feat.flatten(1, -1))
        else:
            raise NotImplementedError(f"num_img={self.num_img} not supported")
        
        # Combine visual and proprioceptive visual_features
        if self.embed_combination_type == 'add' or 'multiply':
            cond_emb = self.cond_embed(torch.cat([visual_feat, prop_emb], dim=-1))
        else:
            cond_emb = torch.cat([visual_feat, prop_emb], dim=-1)
        
        # Combine embeddings based on embed_combination_type
        if self.embed_combination_type == 'add':
            td_cond_emb = td_emb + cond_emb
        elif self.embed_combination_type == 'multiply':
            td_cond_emb = td_emb * cond_emb
        elif self.embed_combination_type == 'concate':
            td_cond_emb = torch.cat([td_emb, cond_emb], dim=-1)
        
        emd=torch.cat([action_embed, td_cond_emb], dim=-1)
        
        # Pass through Transformer blocks
        # omitted
        
        # Final layer to predict velocities
        velocity = self.vel_head(emd)
        if output_embedding:
            return velocity.view(B, Ta, Da), td_emb, cond_emb
        return velocity.view(B, Ta, Da)
    

from model.flow.mlp_flow import NoisyFlowMLP, ExploreNoiseNet
class NoisyShortCutFlowMLP(NoisyFlowMLP):
    def __init__(
        self,
        policy:ShortCutFlowMLP,
        denoising_steps:int,
        learn_explore_noise_from:int,
        inital_noise_scheduler_type:str,
        min_logprob_denoising_std:float,
        max_logprob_denoising_std:float,
        learn_explore_time_embedding:bool,
        time_dim_explore:int,
        use_time_independent_noise:bool,
        device,
        noise_hidden_dims=None,
        activation_type='Tanh',
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
        self.policy:ShortCutFlowMLP
    
    # overload to receive shortcut features 
    def init_exploration_noise_net(self):
        if self.use_time_independent_noise:
            # sigma(s)
            # input dims for the noisy net
            noise_input_dim = self.policy.cond_enc_dim
            # hidden dims for the noisy net
            if not self.noise_hidden_dims:
                self.noise_hidden_dims = [16]
        else:
            if self.learn_explore_time_embedding:
                noise_input_dim = self.time_dim_explore + self.policy.cond_enc_dim
                self.time_embedding_explore = nn.Embedding(num_embeddings=self.denoising_steps, 
                                                       embedding_dim = self.time_dim_explore, 
                                                       device=self.device)
            else:
                # sigma(s,t)
                # input dims for the noisy net
                noise_input_dim = self.policy.td_emb_dim + self.policy.cond_enc_dim
                # hidden dims for the noisy net
                if not self.noise_hidden_dims:
                    self.noise_hidden_dims = [int(np.sqrt(noise_input_dim**2 + self.policy.act_dim_total**2))]
        
        self.explore_noise_net=ExploreNoiseNet(in_dim=noise_input_dim, 
                                                out_dim=self.policy.act_dim_total,
                                                logprob_denoising_std_range=[self.min_logprob_denoising_std, self.max_logprob_denoising_std], 
                                                device=self.device,
                                                hidden_dims=self.noise_hidden_dims,
                                                activation_type=self.noise_activation_type)

    # overload
    def forward(
        self,
        action,
        time,
        dt,
        cond,
        learn_exploration_noise=False,
        step=-1,
        verbose=False,
        **kwargs,
    )->Tuple[Tensor, Tensor]:
        """
        inputs:
            x: (B, Ta, Da)
            time: (B,) floating point in {0,1/2,1/4,1/8,...1/2^n} shortcut flow time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching denoising step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        # WARNING: here you must secure that dt and time matches: time must be a multiple of 1.0 / self.denoising_steps.
        vel, td_emb, cond_emb = self.policy.forward(action, time, dt, cond, output_embedding=True)
        
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
                    noise_feature    = torch.cat([td_emb.detach(), cond_emb], dim=-1)
            
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
            
            if verbose:
                log.info(f"step={step}, learnable noise = {noise_std.mean()}")
        if verbose:
            log.info(f"step={step}, set to learn from {self.learn_explore_noise_from}, will learn exploration noise ? {step >= self.learn_explore_noise_from}, noise_std={noise_std.mean()}require_grad={noise_std.requires_grad}")
        
        return vel, noise_std if learn_exploration_noise else noise_std.detach()

class NoisyVisionShortCutFlowMLP(NoisyShortCutFlowMLP):
    def __init__(
        self,
        policy:ShortCutFlowViT,
        denoising_steps:int,
        learn_explore_noise_from:int,
        inital_noise_scheduler_type:str,
        min_logprob_denoising_std:float,
        max_logprob_denoising_std:float,
        learn_explore_time_embedding:bool,
        time_dim_explore:int,
        use_time_independent_noise:bool,
        device,
        noise_hidden_dims=None,
        activation_type='Tanh',
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
            activation_type,
        )
        self.policy:ShortCutFlowViT
    # overload
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
            time: (B,) floating point in {0,1/2,1/4,1/8,...1/2^n} shortcut flow time
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
            step: (B,) torch.tensor, optional, flow matching inference step, from 0 to denoising_steps-1
            *here, B is the n_envs
        outputs:
             vel                [B, Ta, Da]
             noise_std          [B, Ta x Da]
        """
        B = action.shape[0]
        # this is new for shortcut flows:
        dt = torch.full((B,), 1.0 / self.denoising_stepss, device=self.device)
        # WARNING: here you must secure that dt and time matches: time must be a multiple of 1.0 / self.denoising_steps.
        vel, td_emb, cond_emb = self.policy.forward(action, time, dt, cond, output_embedding=True)
        
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
                    noise_feature    = torch.cat([td_emb.detach(), cond_emb], dim=-1)
            noise_std = self.explore_noise_net.forward(noise_feature=noise_feature)
        return vel, noise_std if learn_exploration_noise else noise_std.detach()
