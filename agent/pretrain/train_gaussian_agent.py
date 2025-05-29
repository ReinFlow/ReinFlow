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

import logging
import wandb
import numpy as np

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device


class TrainGaussianAgent(PreTrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Entropy bonus - not used right now since using fixed_std
        self.ent_coef = cfg.train.get("ent_coef", 0)
    
    def run(self):

        timer = Timer()
        self.epoch = 1
        for _ in range(self.n_epochs):

            # train
            loss_train_epoch = []
            ent_train_epoch = []
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)

                self.model.train()
                loss_train, infos_train = self.model.loss(
                    *batch_train,
                    ent_coef=self.ent_coef,
                )
                loss_train.backward()
                loss_train_epoch.append(loss_train.item())
                ent_train_epoch.append(infos_train["entropy"].item())

                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_train = np.mean(loss_train_epoch)
            ent_train = np.mean(ent_train_epoch)

            # validate
            loss_val_epoch = []
            if self.dataloader_val is not None and self.epoch % self.val_freq == 0:
                self.model.eval()
                for batch_val in self.dataloader_val:
                    if self.dataset_val.device == "cpu":
                        batch_val = batch_to_device(batch_val)
                    loss_val, infos_val = self.model.loss(
                        *batch_val,
                        ent_coef=self.ent_coef,
                    )
                    loss_val_epoch.append(loss_val.item())
                self.model.train()
            loss_val = np.mean(loss_val_epoch) if len(loss_val_epoch) > 0 else None

            # update lr
            self.lr_scheduler.step()

            # update ema
            if self.epoch % self.update_ema_freq == 0:
                self.step_ema()

            # always save the last checkpoint for resume 
            self.save_last_model()
            
            # save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
            
            # log loss
            if self.epoch % self.log_freq == 0:
                infos_str = " | ".join(
                    [f"{key}: {val:8.4f}" for key, val in infos_train.items()]
                )
                log.info(
                    f"{self.epoch}: train loss {loss_train:8.4f} | {infos_str} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    if loss_val is not None:
                        wandb.log(
                            {"loss - val": loss_val}, step=self.epoch, commit=False
                        )
                    wandb.log(
                        {
                            "loss - train": loss_train,
                            "entropy - train": ent_train,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            # count
            self.epoch += 1
