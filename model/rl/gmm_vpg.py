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

import torch
import logging
from model.common.gmm import GMMModel


class VPG_GMM(GMMModel):
    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # Re-name network to actor
        self.actor_ft = actor

        # Value function for obs - simple MLP
        self.critic = critic.to(self.device)

    # ---------- Sampling ----------#

    @torch.no_grad()
    def forward(self, cond, deterministic=False):
        return super().forward(
            cond=cond,
            deterministic=deterministic,
        )

    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        actions,
    ):
        B = len(actions)
        dist, entropy, std = self.forward_train(
            cond,
            deterministic=False,
        )
        log_prob = dist.log_prob(actions.view(B, -1))
        return log_prob, entropy, std

    def loss(self, obs, chains, reward):
        raise NotImplementedError
