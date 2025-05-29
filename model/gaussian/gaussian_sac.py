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
Soft Actor Critic (SAC) with Gaussian policy.
"""

import torch
import logging
from copy import deepcopy
import torch.nn.functional as F

from model.common.gaussian import GaussianModel
from model.common.critic import CriticObsAct
log = logging.getLogger(__name__)


class SAC_Gaussian(GaussianModel):
    def __init__(
        self,
        actor,
        critic:CriticObsAct,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # initialize doubel critic networks
        self.critic: CriticObsAct = critic.to(self.device)
        
        # initialize double target networks
        self.target_critic = deepcopy(self.critic).to(self.device)

    def loss_critic(
        self,
        obs,
        next_obs,
        actions,
        rewards,
        terminated,
        gamma,
        alpha,
    ):
        with torch.no_grad():
            next_actions, next_logprobs = self.forward(
                cond=next_obs,
                deterministic=False,
                get_logprob=True,
            )
            next_q1, next_q2 = self.target_critic(
                next_obs,
                next_actions,
            )
            next_q = torch.min(next_q1, next_q2) - alpha * next_logprobs
            
            # target value
            target_q = rewards + gamma * next_q * (1 - terminated)
        # log.info(f"next_q={next_q.mean().item():.2f}")
        current_q1, current_q2 = self.critic(obs, actions)
        
        loss_critic = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        return loss_critic

    def loss_actor(self, obs, alpha):
        action, logprob = self.forward(
            obs,
            deterministic=False,
            reparameterize=True,
            get_logprob=True,
        )
        current_q1, current_q2 = self.critic(obs, action)
        loss_actor = -torch.min(current_q1, current_q2) + alpha * logprob
        return loss_actor.mean()

    def loss_temperature(self, obs, alpha, target_entropy):
        with torch.no_grad():
            _, logprob = self.forward(
                obs,
                deterministic=False,
                get_logprob=True,
            )
        loss_alpha = -torch.mean(alpha * (logprob + target_entropy))
        return loss_alpha
    
    def loss_temperature_log(self, obs, logalpha, target_entropy):
        with torch.no_grad():
            _, logprob = self.forward(
                obs,
                deterministic=False,
                get_logprob=True,
            )
        loss_alpha = -torch.mean(logalpha * (logprob + target_entropy))
        return loss_alpha

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
