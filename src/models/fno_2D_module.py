from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt


class FNO2dModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        loss: object,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        gamma: float = 0.5,
        step_size: int = 100,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_acc = Accuracy()
        # self.val_acc = Accuracy()
        # self.test_acc = Accuracy()

        # for logging best so far validation accuracy

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, y_hat, y = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, y_hat, y = self.step(batch)

        if (batch_idx in {0, 2, 4}) & (self.global_step % 3) == 0:
            figure = self.plot_predictions(y=y[0,:,:,0], y_hat=y_hat[0,:,:,0])
            self.logger.experiment.add_figure('Network Prediction', figure, global_step=self.global_step)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, y_hat, y = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def plot_predictions(self, y, y_hat):
        fig, axs = plt.subplots(ncols = 3, figsize=(20,10))
        titles = ['y real', 'y_hat_real', 'Difference']
        content = [y, y_hat, (y-y_hat)]
        for i in range(len(titles)):
            axs[i].imshow(content[i].cpu().numpy(), cmap='seismic')
            axs[i].set_title(titles[i])
        return fig
