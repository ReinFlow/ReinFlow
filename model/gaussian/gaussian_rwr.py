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
Reward-weighted regression (RWR) for Gaussian policy.

"""

import torch
import logging
from model.common.gaussian import GaussianModel
import torch.distributions as D

log = logging.getLogger(__name__)


class RWR_Gaussian(GaussianModel):

    def __init__(
        self,
        actor,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)

        # assign actor
        self.actor = self.network

    # override
    def loss(self, actions, obs, reward_weights):
        B = len(obs)
        means, scales = self.network(obs)

        dist = D.Normal(loc=means, scale=scales)
        log_prob = dist.log_prob(actions.view(B, -1)).mean(-1)
        log_prob = log_prob * reward_weights
        log_prob = -log_prob.mean()
        return log_prob

    # override
    @torch.no_grad()
    def forward(self, cond, deterministic=False, **kwargs):
        actions = super().forward(
            cond=cond,
            deterministic=deterministic,
        )
        return actions
