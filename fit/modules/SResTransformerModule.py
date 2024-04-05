from pyexpat import model
import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.modules.loss import _fc_prod_loss, _fc_sum_loss
from fit.transformers_fit.SResTransformer import SResTransformer
from fit.utils.utils import convert2DFT
from fit.utils.PSNR import RangeInvariantPsnr as PSNR
from fit.utils.RAdam import RAdam
import wandb
import numpy as np
import torch.fft
from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi

class SResTransformerModule(LightningModule):
    def __init__(self, img_shape, coords, dst_flatten_order, dst_order,loss='prod',model_type = 'fast',lr=0.0001,weight_decay=0.01,n_layers=4, n_heads=4,
                d_query=32,num_shells=4,attention_dropout=0.1,dropout=0.1,w_phi=1):
        super().__init__()
        
        self.model_type = model_type
        self.coords = coords
        self.dst_flatten_order = dst_flatten_order
        self.dst_order = dst_order
        self.dft_shape = (img_shape, img_shape // 2 + 1)
        self.d_model = n_heads*d_query
        self.n_heads = n_heads
        self.d_query = d_query
        self.shells = num_shells
        self.n_layers = n_layers
        self.loss = loss
        self.w_phi = w_phi

        self.save_hyperparameters("model_type",
                                  "img_shape",
                                  "loss",
                                  "lr",
                                  "weight_decay",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "num_shells",
                                  "attention_dropout",
                                  "dropout")
        
        # Set the loss function based on the input loss type
        if loss == 'prod':
            self.loss = _fc_prod_loss
        elif loss == 'sum':
            self.loss = _fc_sum_loss

        self.outputs = [] #for storing outputs of training step

        # Initialize the SResTransformer Model
        self.sres = SResTransformer(d_model=self.hparams.n_heads * self.hparams.d_query,
                                    coords=self.coords,
                                    flatten_order=self.dst_flatten_order,
                                    attention_type='full',
                                    model_type=self.model_type,
                                    n_layers=self.hparams.n_layers,
                                    n_heads=self.hparams.n_heads,
                                    d_query=self.hparams.d_query,
                                    dropout=self.hparams.dropout,
                                    attention_dropout=self.hparams.attention_dropout)

        x, y = np.meshgrid(range(self.dft_shape[1]), range(-self.dft_shape[0] // 2, self.dft_shape[0] // 2 + 1))
        radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), self.dft_shape[0] // 2 + 1, 0)
        num_shells = self.shells
        self.input_seq_length = np.sum(np.round(radii) < num_shells)

    def forward(self, x):
        return self.sres.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'monitor': 'Validation/avg_val_phi_loss'
        }

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=mag_min,
                                                amp_max=mag_max,w_phi = 1)
        return fc_loss, amp_loss, phi_loss

    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch #378,2
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]
        
        pred = self.sres.forward(x_fc)

        fc_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)

        self.log_dict({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss},prog_bar=True,on_step=True)
        self.outputs.append({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss})
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}
    
    def on_train_epoch_end(self):        
        loss = torch.mean(torch.tensor([x['loss'] for x in self.outputs]))
        amp_loss = torch.mean(torch.tensor([x['amp_loss'] for x in self.outputs]))
        phi_loss = torch.mean(torch.tensor([x['phi_loss'] for x in self.outputs]))
        self.log('Train/train_mean_epoch_loss', loss, logger=True, on_epoch=True)
        self.log('Train/train_mean_epoch_amp_loss', amp_loss, logger=True, on_epoch=True)
        self.log('Train/train_mean_epoch_phi_loss', phi_loss, logger=True, on_epoch=True)
        self.outputs = []
        
    def log_val_images(self, fc, mag_min, mag_max):
        lowres, pred, gt = self.predict_and_get_lowres_pred_gt(fc, mag_min=mag_min, mag_max=mag_max)
        i = 0
        lowres_ = torch.clamp((lowres[i].unsqueeze(0) - lowres.min()) / (lowres.max() - lowres.min()), 0, 1)
        pred_ = torch.clamp((pred[i].unsqueeze(0) - pred.min()) / (pred.max() - pred.min()), 0, 1)
        self.logger.experiment.log({f"Validation_Images/val_input_image":[wandb.Image(lowres_.cpu(), caption=f"inputs/img_{i}")],"global_step": self.trainer.global_step})
        self.logger.experiment.log({f"Validation_Images/val_pred_image":[wandb.Image(pred_.cpu(), caption=f"predictions/img_{i}")],"global_step": self.trainer.global_step})   

    def validation_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]
        pred = self.sres.forward(x_fc)
    
        val_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)

        if self.current_epoch%20 == 0 and batch_idx == 0 and self.logger._name != 'lightning_logs':
            self.save_forward_func_output(pred,mag_min, mag_max)
            self.log_val_images(fc, mag_min, mag_max)
        self.val_outputs = {'val_loss': val_loss, 'val_amp_loss': amp_loss, 'val_phi_loss': phi_loss}
        self.log_dict(self.val_outputs)
        return self.val_outputs

    def save_forward_func_output(self, pred,mag_min, mag_max):
        pred_img = self.convert2img(pred, mag_min, mag_max)
        self.logger.experiment.log({f"Validation_Images/val_fwd_fnc_output":[wandb.Image(pred_img[0].cpu(), caption=f"pred_of_forward_menthod")],"global_step": self.trainer.global_step})

    def on_validation_epoch_end(self):
        val_loss = self.val_outputs['val_loss']
        amp_loss = self.val_outputs['val_amp_loss']
        phi_loss = self.val_outputs['val_phi_loss']

        self.log('Validation/avg_val_loss', torch.mean(val_loss), logger=True, on_epoch=True)
        self.log('Validation/avg_val_amp_loss', torch.mean(amp_loss), logger=True, on_epoch=True)
        self.log('Validation/avg_val_phi_loss', torch.mean(phi_loss), logger=True, on_epoch=True)
    
    
    def on_test_epoch_end(self):
        lowres_psnrs = torch.stack(self.test_outputs[0])
        pred_psnrs = torch.stack(self.test_outputs[1])
        self.log('Test/Input Mean PSNR', torch.mean(lowres_psnrs), logger=True,on_epoch=True)
        self.log('Test/Input SEM PSNR', torch.std(lowres_psnrs / np.sqrt(len(lowres_psnrs))),logger=True,on_epoch=True)
        self.log('Test/Prediction Mean PSNR', torch.mean(pred_psnrs), logger=True,on_epoch=True)
        self.log('Test/Prediction SEM PSNR', torch.std(pred_psnrs / np.sqrt(len(pred_psnrs))),logger=True,on_epoch=True)


    def test_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        lowres_img,pred_img,gt_img = self.predict_and_get_lowres_pred_gt(fc, mag_min, mag_max)
        lowres_psnr = PSNR(gt_img,lowres_img)
        pred_psnr = PSNR(gt_img,pred_img)
        self.test_outputs = [lowres_psnr, pred_psnr, pred_img, lowres_img, gt_img]
        return (lowres_psnr, pred_psnr)
    
    def on_test_epoch_end(self):
        lowres_psnrs = self.test_outputs[0]
        pred_psnrs = self.test_outputs[1]
        self.log('Input Mean PSNR', torch.mean(lowres_psnrs), logger=True)
        self.log('Input SEM PSNR', torch.std(lowres_psnrs / np.sqrt(len(lowres_psnrs))),logger=True)
        self.log('Prediction Mean PSNR', torch.mean(pred_psnrs), logger=True)
        self.log('Prediction SEM PSNR', torch.std(pred_psnrs / np.sqrt(len(pred_psnrs))),logger=True)
    
    def convert2img(self, fc, mag_min, mag_max):
        dft = convert2DFT(x=fc, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=self.dst_flatten_order,
                            img_shape=self.hparams.img_shape)
        return torch.fft.irfftn(dft, s = 2 * (self.hparams.img_shape,), dim=[1,2])                                

    def predict_and_get_lowres_pred_gt(self, fc, mag_min, mag_max):
        x_fc = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        pred = self.sres.forward_inference(x_fc, self.input_seq_length)
        lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc, pred, mag_min, mag_max)
        return lowres_img, pred_img, gt_img

    def get_lowres_pred_gt(self, fc, pred, mag_min, mag_max):
        pred_img = self.convert2img(pred, mag_min=mag_min, mag_max=mag_max) #no need for [dst_flatten_order] as x_fc was made that way
        lowres = torch.zeros_like(pred)
        lowres += fc.min()
        lowres[:, :self.input_seq_length] = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        lowres_img = self.convert2img(fc=lowres, mag_min=mag_min, mag_max=mag_max)
        gt_img = self.convert2img(fc=fc[:, self.dst_flatten_order], mag_min=mag_min, mag_max=mag_max)
        lowres_img_denormed = denormalize(lowres_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img_denormed = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        gt_img_denormed = denormalize(gt_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)

        return lowres_img_denormed, pred_img_denormed, gt_img_denormed