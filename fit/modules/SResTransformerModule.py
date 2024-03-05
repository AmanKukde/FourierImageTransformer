import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.modules.loss import _fc_prod_loss, _fc_sum_loss
from fit.transformers.SResTransformer import SResTransformerTrain
from fit.utils import denormalize_FC, convert2DFT#, PSNR
from fit.transformers.PSNR import RangeInvariantPsnr as PSNR
from fit.utils.RAdam import RAdam
import wandb
import numpy as np

import torch.fft

from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi

class SResTransformerModule(LightningModule):
    def __init__(self, d_model, img_shape,
                 coords, dst_flatten_order, dst_order,
                 loss='prod',
                 mode='train',
                 lr=0.0001,
                 weight_decay=0.01,\
                 n_layers=4, n_heads=4, d_query=32, dropout=0.1, attention_dropout=0.1,num_shells=4,model_path = None,saved_model_path = None):
        super().__init__()
        self.outputs = []
        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "loss",
                                  "lr",
                                  "weight_decay",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout",
                                  "num_shells")

        self.coords = coords
        self.dst_flatten_order = dst_flatten_order
        self.dst_order = dst_order
        self.dft_shape = (img_shape, img_shape // 2 + 1)
        self.shells = num_shells
        
        self.coords = coords
        self.dst_flatten_order = dst_flatten_order
        self.dst_order = dst_order
        self.dft_shape = (img_shape, img_shape // 2 + 1)

        if loss == 'prod':
            self.loss = _fc_prod_loss
        else:
            self.loss = _fc_sum_loss

        self.sres = SResTransformerTrain(d_model=self.hparams.d_model,
                                         coords=self.coords,
                                         flatten_order=self.dst_flatten_order,
                                         attention_type='full',
                                         n_layers=self.hparams.n_layers,
                                         n_heads=self.hparams.n_heads,
                                         d_query=self.hparams.d_query,
                                         dropout=self.hparams.dropout,
                                         attention_dropout=self.hparams.attention_dropout)
        self.sres_pred = None
        x, y = np.meshgrid(range(self.dft_shape[1]), range(-self.dft_shape[0] // 2, self.dft_shape[0] // 2 + 1))
        radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), self.dft_shape[0] // 2 + 1, 0)
        num_shells = 8
        self.input_seq_length = np.sum(np.round(radii) < num_shells)

    def forward(self, x):
        return self.sres.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Train/train_loss'
        }

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=mag_min,
                                                amp_max=mag_max)
        return fc_loss, amp_loss, phi_loss

    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_order]

        pred = self.sres.forward(x_fc)

        fc_loss, amp_loss, phi_loss = self.criterion(pred, x_fc, mag_min, mag_max)
        self.outputs = [{'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}]
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def on_train_epoch_end(self):        
        loss = torch.mean(torch.tensor([x['loss'] for x in self.outputs]))
        amp_loss = torch.mean(torch.tensor([x['amp_loss'] for x in self.outputs]))
        phi_loss = torch.mean(torch.tensor([x['phi_loss'] for x in self.outputs]))
        self.log('Train/train_loss', loss, logger=True, on_epoch=True)
        self.log('Train/train_amp_loss', amp_loss, logger=True, on_epoch=True)
        self.log('Train/train_phi_loss', phi_loss, logger=True, on_epoch=True)
        self.outputs = []

