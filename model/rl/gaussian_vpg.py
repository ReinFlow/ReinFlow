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
Policy gradient for Gaussian policy

"""

import torch
from copy import deepcopy
import logging
from model.common.gaussian import GaussianModel


class VPG_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)

        # Re-name network to actor
        self.actor_ft = actor

        # Save a copy of original actor
        self.actor = deepcopy(actor)
        for param in self.actor.parameters():
            param.requires_grad = False

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        use_base_policy=False,
    ):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
            network_override=self.actor if use_base_policy else None,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
        use_base_policy=False,
    ):
        B = len(actions)
        dist = self.forward_train(
            cond,
            deterministic=False,
            network_override=self.actor if use_base_policy else None,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        log_prob = log_prob.mean(-1)
        entropy = dist.entropy().mean()
        std = dist.scale.mean()
        return log_prob, entropy, std

    def loss(self, obs, actions, reward):
        raise NotImplementedError
