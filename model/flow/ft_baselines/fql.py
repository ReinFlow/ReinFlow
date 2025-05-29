# The MIT License (MIT)
#
# Copyright (c) 2025 FQL Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Note: the original flow-q-learning repository was implemented in JAX language, reference: https://github.com/seohongpark/fql
# The PyTorch version provided here was translated by ReinFlow Authors on 4/23/2025.

import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import copy
from model.flow.reflow import ReFlow
from model.common.critic import CriticObsAct

class OneStepActor(nn.Module):
    """Distill a multistep flow model to single step stochastic policy"""
    def __init__(self, 
                 obs_dim, 
                 cond_steps,
                 action_dim, 
                 horizon_steps, 
                 hidden_dim=512):
        """Initialize the OneStepActor with a neural network to map observations and noise to actions.

        Args:
            obs_dim (int): Dimension of the observation space.
            cond_steps (int): Number of observation steps in the horizon.
            action_dim (int): Dimension of the action space.
            horizon_steps (int): Number of action steps in the horizon.
            hidden_dim (int, optional): Hidden layer size. Defaults to 512.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.cond_steps=cond_steps
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.net = nn.Sequential(
            nn.Linear(cond_steps*obs_dim + horizon_steps * action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon_steps * action_dim),
        )

    def forward(self, cond: Dict[str, Tensor], z: Tensor) -> Tensor:
        """Generate actions from observations and noise using the actor network.

        Args:
            cond (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state', which is a tensor of shape (batch, cond_steps, obs_dim)
            z (Tensor): Noise tensor of shape (batch, horizon_steps, action_dim).

        Returns:
            Tensor: Actions of shape (batch, horizon_steps, action_dim).
        """
        s = cond['state']       # (batch, cond_step, obs_dim)
        B = s.shape[0]
        s_flat = s.view(B, -1)  # (batch, cond_step * obs_dim)
        z_flat = z.view(B, -1)  # (batch, horizon*action_dim)
        feature = torch.cat([s_flat, z_flat], dim=-1)
        output = self.net(feature)
        actions: Tensor = output.view(B, self.horizon_steps, self.action_dim)
        return actions

class FQLModel(nn.Module):
    def __init__(self, 
                 bc_flow: ReFlow, 
                 actor: OneStepActor, 
                 critic: CriticObsAct, 
                 inference_steps: int,
                 normalize_q_loss: bool,
                 device):
        """Initialize the FQL model with behavior cloning flow, actor, critic, and target critic.

        Args:
            bc_flow (ReFlow): Behavior cloning flow model.
            actor (OneStepActor): Actor network for action generation.
            critic (CriticObsAct): Critic network for Q-value estimation.
            inference_steps (int): Number of inference steps for the flow model.
            normalize_q_loss (bool): Whether to normalize the Q-value loss.
            device: Device to run the model on (e.g., 'cuda').
        """
        super().__init__()
        self.bc_flow:ReFlow = bc_flow
        self.actor:OneStepActor = actor
        self.critic:CriticObsAct = critic
        self.target_critic:CriticObsAct = copy.deepcopy(self.critic)
        self.device = device
        self.inference_steps = inference_steps # for the base model.
        self.normalize_q_loss=normalize_q_loss
        
    def forward(self, 
                cond: Dict[str, Tensor], 
                mode:str='onestep'):
        """Generate actions for a batch of observations using the actor with random noise.

        Args:
            cond (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state'.

        Returns:
            Tensor: Actions of shape (batch, horizon_steps, action_dim).
        """
        batch_size = cond['state'].shape[0]
        assert mode in ['onestep', 'base_model']
        if mode=='onestep':
            z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
            actions = self.actor(cond, z)
        elif mode=='base_model': 
            actions = self.bc_flow.sample(cond, self.inference_steps, record_intermediate=False, clip_intermediate_actions=False).trajectories
        return actions 
    
    def loss_bc_flow(self, obs: Dict[str, Tensor], actions: Tensor):
        """Compute the behavior cloning flow loss by comparing predicted and target flow values. This is the same as doing pre-training.

        Args:
            obs (Dict[str, Tensor]): Dictionary containing the state tensor with key 'state'.
            actions (Tensor): Ground truth actions of shape (batch, horizon_steps, action_dim).

        Returns:
            Tensor: Mean squared error loss for the behavior cloning flow.
        """
        (xt, t), v = self.bc_flow.generate_target(actions)
        v_hat = self.bc_flow.network(xt, t, obs)
        loss_bc_flow = F.mse_loss(v_hat, v)
        return loss_bc_flow

    def loss_critic(self, obs, actions, next_obs, rewards, terminated, gamma)->Tuple[Tensor, Dict]:
        """Compute the critic loss using mean squared error between predicted and target Q-values, with debug information.
        Use the one-step actor's action to compute q target, use dataset batch's action as q  prediction. Then adopt TD loss.
        Args:
            obs (Dict[str, Tensor]): Current state observations.
            actions (Tensor): Actions taken.
            next_obs (Dict[str, Tensor]): Next state observations.
            rewards (Tensor): Rewards received.
            terminated (Tensor): Termination flags.
            gamma (float): Discount factor.

        Returns:
            Tuple[Tensor, Dict]: Critic loss and dictionary of debug information.
        """
        with torch.no_grad():
            # get action from one-step actor.
            z = torch.randn_like(actions, device=self.device)
            next_actions = self.actor.forward(next_obs, z)
            next_actions.clamp_(-1,1)
            next_q1, next_q2 = self.target_critic.forward(next_obs, next_actions)
            next_q = torch.mean(torch.stack([next_q1, next_q2], dim=0), dim=0)   # this is typical of fql. possible bug in reproduction
            # target q value
            target = rewards + gamma * (1 - terminated) * next_q
        # get q prediction from dataset actions.
        q1, q2 = self.critic.forward(obs, actions)
        loss_critic = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        # debug info
        loss_critic_info={
            'loss_critic': loss_critic.item(),
            'q1_mean': q1.mean().item(),
            'q1_max': q1.max().item(),
            'q1_min': q1.min().item(),
            'q2_mean': q2.mean().item(),
            'q2_max': q2.max().item(),
            'q2_min': q2.min().item(),
            'loss_critic_1': F.mse_loss(q1, target).item(),
            'loss_critic_2': F.mse_loss(q2, target).item(),  # Fixed from q1 to q2
        }
        return loss_critic, loss_critic_info

    def loss_actor(self, obs:Dict[str, Tensor], action_batch:Tensor, alpha:float)->Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the actor loss combining behavior cloning, Q-value, and distillation losses, with debug information.

        Args:
            obs (Dict[str, Tensor]): State observations.
            action_batch (Tensor): Ground truth actions for behavior cloning.
            alpha (float): Weight for the distillation loss.

        Returns:
            Tuple[Tensor, Dict]: Actor loss and dictionary of debug information.
        """
        # get distillation loss
        batch_size = obs['state'].shape[0]
        z = torch.randn(batch_size, self.actor.horizon_steps, self.actor.action_dim, device=self.device)
        a_ω = self.actor.forward(obs, z)
        a_θ = self.bc_flow.sample(cond=obs, inference_steps=self.inference_steps, record_intermediate=False, clip_intermediate_actions=False, z=z).trajectories  # Use same z
        distill_loss = F.mse_loss(a_ω, a_θ)
        
        # get q loss
        actor_actions = torch.clamp(a_ω, -1, 1)
        q1, q2 = self.critic.forward(obs, actor_actions)
        q = torch.mean(torch.stack([q1, q2], dim=0), dim=0)
        q_loss = -q.mean()
        if self.normalize_q_loss:
            lam = 1 / torch.abs(q).mean().detach()
            q_loss = lam * q_loss
        
        # get BC loss 
        loss_bc_flow = self.loss_bc_flow(obs, action_batch)
        
        loss_actor = loss_bc_flow + alpha * distill_loss + q_loss

        # debug info
        onestep_expert_bc_loss=F.mse_loss(a_ω, action_batch)
        loss_actor_info={
            'loss_actor': loss_actor.item(),
            'loss_bc_flow': loss_bc_flow.item(),
            'q_loss': q_loss.item(),
            'distill_loss': distill_loss.item(),
            'q': q.mean().item(),
            'onestep_expert_bc_loss': onestep_expert_bc_loss.item()
        }
        return loss_actor, loss_actor_info

    def update_target_critic(self, tau:float):
        """Update the target double critic network."""
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                source_param.data * tau + target_param.data * (1.0 - tau)
            )
                
