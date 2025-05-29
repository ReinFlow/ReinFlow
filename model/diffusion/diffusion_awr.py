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
Advantage-weighted regression (AWR) for diffusion policy.

"""

import logging
import torch

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion


class AWRDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.critic = critic.to(self.device)

        # assign actor
        self.actor = self.network

    def loss_critic(self, obs, advantages):
        # get advantage
        adv = self.critic(obs)

        # Update critic
        loss_critic = torch.mean((adv - advantages) ** 2)
        return loss_critic
