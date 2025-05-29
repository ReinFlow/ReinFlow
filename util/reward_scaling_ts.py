# MIT License

# Copyright (c) 2024 Intelligent Robot Motion Lab

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
# 

# ReinFlow Authors revised from this script


import torch

class RunningMeanStdTensor:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        super().__init__()
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        x = x.to(self.mean.device)  # Ensure x is on the same device as mean
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)  # Use unbiased=False to avoid Bessel's correction
        batch_count = x.size(0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / (tot_count - 1)
        self.count = tot_count


class RunningRewardScalerTensor:
    def __init__(self, num_envs, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False, device=None):
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStdTensor(shape=ret_rms_shape, device=device)
        self.cliprew = cliprew
        self.ret = torch.zeros(num_envs, device=device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        reward = reward.to(self.ret.device)  # Ensure reward is on the same device as ret
        first = first.to(self.ret.device)     # Ensure first is on the same device
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward, first=first, gamma=self.gamma
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets.reshape(-1) if not self.per_env else rets)
        return self.transform(reward)

    def transform(self, reward):
        return torch.clamp(
            reward / torch.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


def backward_discounted_sum(prevret, reward, first, gamma):
    assert first.ndim == 2
    _, nstep = reward.shape
    ret = torch.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret
