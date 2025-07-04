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
GMM policy parameterization.

"""

import torch
import torch.distributions as D
import logging

log = logging.getLogger(__name__)


class GMMModel(torch.nn.Module):

    def __init__(
        self,
        network,
        horizon_steps,
        network_path=None,
        device="cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.network = network.to(device)
        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            self.load_state_dict(
                checkpoint["model"],
                strict=False,
            )
            logging.info("Loaded actor from %s", network_path)
        log.info(
            f"Number of network parameters: {sum(p.numel() for p in self.parameters())}"
        )
        self.horizon_steps = horizon_steps

    def loss(
        self,
        true_action,
        cond,
        **kwargs,
    ):
        B = len(true_action)
        dist, entropy, _ = self.forward_train(
            cond,
            deterministic=False,
        )
        true_action = true_action.view(B, -1)
        loss = -dist.log_prob(true_action)  # [B]
        loss = loss.mean()
        return loss, {"entropy": entropy}

    def forward_train(
        self,
        cond,
        deterministic=False,
    ):
        """
        Calls the MLP to compute the mean, scale, and logits of the GMM. Returns the torch.Distribution object.
        """
        means, scales, logits = self.network(cond)
        if deterministic:
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4

        # mixture components - make sure that `batch_shape` for the distribution is equal to (batch_size, num_modes) since MixtureSameFamily expects this shape
        # Each mode has mean vector of dim T*D
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        component_entropy = component_distribution.entropy()
        approx_entropy = torch.mean(
            torch.sum(logits.softmax(-1) * component_entropy, dim=-1)
        )
        std = torch.mean(torch.sum(logits.softmax(-1) * scales.mean(-1), dim=-1))

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)
        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        return dist, approx_entropy, std

    def forward(self, cond, deterministic=False):
        B = len(cond["state"]) if "state" in cond else len(cond["rgb"])
        T = self.horizon_steps
        dist, _, _ = self.forward_train(
            cond,
            deterministic=deterministic,
        )
        sampled_action = dist.sample()
        sampled_action = sampled_action.view(B, T, -1)
        return sampled_action
