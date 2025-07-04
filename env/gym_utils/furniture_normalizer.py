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
Normalization for Furniture-Bench environments.

TODO: use this normalizer for all benchmarks.

"""

import torch.nn as nn


class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stats = nn.ParameterDict()

    def fit(self, data_dict):
        for key, tensor in data_dict.items():
            min_value = tensor.min(dim=0)[0]
            max_value = tensor.max(dim=0)[0]

            # Check if any column has only one value throughout
            diff = max_value - min_value
            constant_columns = diff == 0

            # Set a small range for constant columns to avoid division by zero
            min_value[constant_columns] -= 1
            max_value[constant_columns] += 1

            self.stats[key] = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_value, requires_grad=False),
                    "max": nn.Parameter(max_value, requires_grad=False),
                },
            )
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def forward(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def _turn_off_gradients(self):
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    def load_state_dict(self, state_dict):

        stats = nn.ParameterDict()
        for key, value in state_dict.items():
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = nn.ParameterDict()
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = nn.Parameter(value)

        self.stats = stats
        self._turn_off_gradients()

        return f"<Added keys {self.stats.keys()} to the normalizer.>"

    def keys(self):
        return self.stats.keys()
