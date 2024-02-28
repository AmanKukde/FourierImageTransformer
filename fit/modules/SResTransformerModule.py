import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from yaml import TagToken
from fit.transformers.PositionalEncoding2D import PositionalEncoding2D

from fit.modules.loss import _fc_prod_loss, _fc_sum_loss,_fc_sum_loss_modified,_fc_prod_loss_modified
from fit.transformers.SResTransformer import SResTransformerTrain, SResTransformerPredict
from fit.utils import denormalize_FC, PSNR, convert2DFT
from fit.utils.RAdam import RAdam
from torch.optim import Adam
import wandb
import numpy as np
import matplotlib.pyplot as plt
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
        self.d_model = n_heads*d_query
        self.n_heads = n_heads
        self.d_query = d_query
        self.mode = mode
        self.shells = num_shells
        self.model_path = model_path
        self.saved_model_path = saved_model_path
        self.n_layers = n_layers
        

        if loss == 'prod':
            self.loss = _fc_prod_loss
        elif loss == 'sum':
            self.loss = _fc_sum_loss
        elif loss == 'prod_modified':
            self.loss = _fc_prod_loss_modified
        elif loss == 'sum_modified':
            self.loss = _fc_sum_loss_modified

        
        self.sres = torch.nn.Transformer(d_model = self.d_model,nhead=8, num_encoder_layers=8,batch_first = True) 

        x, y = np.meshgrid(range(self.dft_shape[1]), range(-self.dft_shape[0] // 2, self.dft_shape[0] // 2 + 1))
        radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), self.dft_shape[0] // 2 + 1, 0)
        num_shells = self.shells
        self.input_seq_length = np.sum(np.round(radii) < num_shells)


        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2) #shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
                d_model // 2, #F/2
                coords=coords,#(r,phi)
                flatten_order=self.dst_flatten_order,
                persistent=False
            ) 


    def forward(self, x):
        return self.sres.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {
            'optimizer': optimizer,
            'monitor': 'loss'
            'monitor': 'loss'
        }

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=mag_min,
                                                amp_max=mag_max)
        return fc_loss, amp_loss, phi_loss
    
    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch

        fc_ = fc[:, self.dst_flatten_order]
        fc_ = self.fourier_coefficient_embedding(fc)
        fc_ = self.pos_embedding(fc_)
 
        src = fc_[:, :self.input_seq_length]
        sos = torch.zeros([32, 1, 256],device = src.device)
        src = torch.cat([sos,src],dim = 1)
        tgt = torch.cat([sos,fc_],dim = 1)
        tgt_mask = torch.ones(tgt.shape[1],tgt.shape[1],device = src.device).triu().T
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0,float('-inf'))
        pred = self.sres.forward(src,tgt,tgt_mask=tgt_mask)
        pred = self.get_amp_phi_from_emb(pred)
        fc_loss, amp_loss, phi_loss = self.criterion(pred, torch.cat([self.get_amp_phi_from_emb(sos),fc],1), mag_min, mag_max)
        self.outputs.append({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss})
        self.log_dict({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss},prog_bar=True,on_step=True)
        self.log_dict({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss},prog_bar=True,on_step=True)
        
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}
    
    def on_train_epoch_end(self):        
        loss = torch.mean(torch.tensor([x['loss'] for x in self.outputs]))
        amp_loss = torch.mean(torch.tensor([x['amp_loss'] for x in self.outputs]))
        phi_loss = torch.mean(torch.tensor([x['phi_loss'] for x in self.outputs]))
        self.log('Train/train_loss', loss, logger=True, on_epoch=True)
        self.log('Train/train_amp_loss', amp_loss, logger=True, on_epoch=True)
        self.log('Train/train_phi_loss', phi_loss, logger=True, on_epoch=True)
        self.log('Train/train_loss', loss, logger=True, on_epoch=True)
        self.log('Train/train_amp_loss', amp_loss, logger=True, on_epoch=True)
        self.log('Train/train_phi_loss', phi_loss, logger=True, on_epoch=True)
        self.outputs = []
        
    def get_amp_phi_from_emb(self, emb):
        amp = torch.tanh(torch.nn.Linear(emb.shape[-1], 1,device = emb.device)(emb))
        phi = torch.tanh(torch.nn.Linear(emb.shape[-1], 1,device = emb.device)(emb))
        return torch.cat([amp, phi], dim=-1)

    def validation_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch

        fc_ = fc[:, self.dst_flatten_order]
        fc_ = self.fourier_coefficient_embedding(fc)
        fc_ = self.pos_embedding(fc_)
 
        src = fc_[:, :self.input_seq_length]
        sos = torch.zeros([32, 1, 256],device = src.device)
        src = torch.cat([sos,src],dim = 1)
        tgt = torch.cat([sos,fc_],dim = 1)
        t = self.sres(src,sos)
        for _ in range(378):
            t = self.sres(src,torch.cat([sos,t],dim = 1))
        pred = t
        pred = self.get_amp_phi_from_emb(pred)
        fc_loss, amp_loss, phi_loss = self.criterion(pred, tgt, mag_min, mag_max)
        self.outputs.append({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss})
        self.log_dict({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss},prog_bar=True,on_step=True)
        self.log_images(fc, pred[:,1:], mag_min, mag_max)
        return {'val_loss': fc_loss, 'val_amp_loss': amp_loss, 'val_phi_loss': phi_loss}

    def log_images(self, fc, pred,mag_min, mag_max):
        pred[:,:self.input_seq_length] = fc[:,:self.input_seq_length]
        lowres, pred, gt = self.get_lowres_pred_gt(fc, pred,mag_min=mag_min, mag_max=mag_max)
        for i in range(min(3, len(lowres))):
            lowres_ = torch.clamp((lowres[i].unsqueeze(0) - lowres.min()) / (lowres.max() - lowres.min()), 0, 1)
            pred_ = torch.clamp((pred[i].unsqueeze(0) - pred.min()) / (pred.max() - pred.min()), 0, 1)
            gt_ = torch.clamp((gt[i].unsqueeze(0) - gt.min()) / (gt.max() - gt.min()), 0, 1)

            self.logger.experiment.log({f"Validation_Images/val_input_image":[wandb.Image(lowres_.cpu(), caption=f"inputs/img_{i}")],"global_step": self.trainer.global_step})
            self.logger.experiment.log({f"Validation_Images/val_pred_image":[wandb.Image(pred_.cpu(), caption=f"predictions/img_{i}")],"global_step": self.trainer.global_step})                                   
            self.logger.experiment.log({f"Validation_Images/val_gt_image":[wandb.Image(gt_.cpu(), caption=f"ground_truth/img_{i}")],"global_step": self.trainer.global_step})

    # def convert2img(self, fc, mag_min, mag_max):
    #     dft = convert2DFT(x=fc, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=self.dst_flatten_order,
    #                       img_shape=self.hparams.img_shape)
    #     return torch.fft.irfftn(dft, s = 2 * (self.hparams.img_shape,), dim=[1,2])

    def get_lowres_pred_gt(self, fc, pred, mag_min, mag_max):
        pred_img = self.convert2img(fc=pred, mag_min=mag_min, mag_max=mag_max)
        lowres = torch.zeros_like(pred)
        lowres += fc.min()
        lowres[:, :self.input_seq_length] = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        lowres_img = self.convert2img(fc=lowres, mag_min=mag_min, mag_max=mag_max)
        gt_img = self.convert2img(fc=fc[:, self.dst_flatten_order], mag_min=mag_min, mag_max=mag_max)
        return lowres_img, pred_img, gt_img
