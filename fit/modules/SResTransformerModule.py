import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


from fit.modules.loss import _fc_prod_loss, _fc_sum_loss,_fc_sum_loss_modified,_fc_prod_loss_modified
from fit.transformers.SResTransformer import SResTransformerTrain
from fit.utils import denormalize_FC, PSNR, convert2DFT
from fit.transformers.PSNR import RangeInvariantPsnr as PSNR
from fit.utils.RAdam import RAdam
import wandb
import numpy as np
# from fit.transformers.PSNR import RangeInvariantPsnr as PSNR
import torch.fft
from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi
import matplotlib.pyplot as plt


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
        self.loss = loss
        

        # #self.logger = WandbLogger(save_dir='/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/examples/models/sres') 

        if loss == 'prod':
            self.loss = _fc_prod_loss
        elif loss == 'sum':
            self.loss = _fc_sum_loss
        elif loss == 'prod_modified':
            self.loss = _fc_prod_loss_modified
        elif loss == 'sum_modified':
            self.loss = _fc_sum_loss_modified
        self.sres = SResTransformerTrain(d_model=self.d_model,
                                         coords=self.coords,
                                         flatten_order=self.dst_flatten_order,
                                         n_layers=self.n_layers,
                                         n_heads=self.n_heads,
                                         d_query=self.d_query,
                                         dropout= 0.1,
                                         attention_dropout=0.1)
        if len(self.model_path) > 0:
            weights = torch.load(self.model_path)
            sd = {}
            for k in weights['state_dict'].keys():
                if k[:5] == 'sres.':
                    sd[k[5:]] = weights['state_dict'][k]
            self.sres.load_state_dict(sd)
            self.sres.to(self.device)
            
            print("weights loaded to train model successfully")

        # self.sres_pred = None
        x, y = np.meshgrid(range(self.dft_shape[1]), range(-self.dft_shape[0] // 2, self.dft_shape[0] // 2 + 1))
        radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), self.dft_shape[0] // 2 + 1, 0)
        num_shells = self.shells
        self.input_seq_length = np.sum(np.round(radii) < num_shells)

    def forward(self, x):
        return self.sres.forward(x)

    def configure_optimizers(self):
        # optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        optimizer = torch.optim.Adam(self.sres.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
            'monitor': 'Train/train_loss'
        }

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=mag_min,
                                                amp_max=mag_max)
        return fc_loss, amp_loss, phi_loss

    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]
        pred = self.sres.forward(x_fc,y_fc)

        
        if self.loss == 'mse':
            lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc=fc,pred = pred, mag_min=mag_min, mag_max=mag_max)
            mse_loss = torch.mean(torch.pow((pred_img - gt_img),2))
            self.outputs.append({'loss': mse_loss})
            self.log_dict({'loss': mse_loss},prog_bar=True,on_step=True)
            return {'loss': mse_loss}
        
        if self.loss == 'cce':
            lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc=fc,pred = pred, mag_min=mag_min, mag_max=mag_max)
            loss_func = torch.nn.CrossEntropyLoss()
            cce_loss = loss_func(pred_img,gt_img)
            self.outputs.append({'loss': cce_loss})
            self.log_dict({'loss': cce_loss},prog_bar=True,on_step=True)
            return {'loss': cce_loss}

        fc_loss, amp_loss, phi_loss = self.criterion(pred,y_fc , mag_min, mag_max)

        self.outputs.append({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss})
        self.log_dict({'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss},prog_bar=True,on_step=True)
        
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}
    
    def on_train_epoch_end(self):        
        loss = torch.mean(torch.tensor([x['loss'] for x in self.outputs]))
        amp_loss = torch.mean(torch.tensor([x['amp_loss'] for x in self.outputs]))
        phi_loss = torch.mean(torch.tensor([x['phi_loss'] for x in self.outputs]))
        self.log('Train/train_loss', loss, logger=True, on_epoch=True)
        self.log('Train/train_amp_loss', amp_loss, logger=True, on_epoch=True)
        self.log('Train/train_phi_loss', phi_loss, logger=True, on_epoch=True)
        self.outputs = []
        

    def validation_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]

        pred = self.sres.forward_i(x_fc)

        val_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)
        if batch_idx == 0:
            self.log_val_images(fc, mag_min, mag_max)
        
        lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc=fc, mag_min=mag_min, mag_max=mag_max)
        lowres_img = denormalize(lowres_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        gt_img = denormalize(gt_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)

        lowres_vs_gt_psnr = torch.mean(torch.tensor([PSNR(gt_img[i], lowres_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                       range(gt_img.shape[0])]))
        pred_vs_gt_psnr = torch.mean(torch.tensor([PSNR(gt_img[i], pred_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                       range(gt_img.shape[0])]))
        low_res_vs_pred_psnr = torch.mean(torch.tensor([PSNR(lowres_img[i], pred_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                     range(gt_img.shape[0])]))
        
        
        self.val_outputs = {'val_loss': val_loss, 'val_amp_loss': amp_loss, 'val_phi_loss': phi_loss, 'val_lowres_psnr': lowres_vs_gt_psnr, 'val_pred_psnr': pred_vs_gt_psnr, 'val_lowres_vs_pred_psnr': low_res_vs_pred_psnr}
        #self.log_dict(self.val_outputs)
        return self.val_outputs

    def log_val_images(self, fc, mag_min, mag_max):
        self.load_test_model(self.trainer.checkpoint_callback.last_model_path)
        lowres, pred, gt = self.get_lowres_pred_gt(fc, mag_min=mag_min, mag_max=mag_max)
        for i in range(min(3, len(lowres))):
            lowres_ = torch.clamp((lowres[i].unsqueeze(0) - lowres.min()) / (lowres.max() - lowres.min()), 0, 1)
            pred_ = torch.clamp((pred[i].unsqueeze(0) - pred.min()) / (pred.max() - pred.min()), 0, 1)
            gt_ = torch.clamp((gt[i].unsqueeze(0) - gt.min()) / (gt.max() - gt.min()), 0, 1)

            plt.imshow(torch.cat([lowres_, pred_, gt_], dim=2).squeeze().cpu().numpy(), cmap='gray');
            plt.savefig(f'val_{i}.png')

    def on_validation_epoch_end(self):
        val_loss = self.val_outputs['val_loss']
        amp_loss = self.val_outputs['val_amp_loss']
        phi_loss = self.val_outputs['val_phi_loss']

        self.log('Validation/avg_val_loss', torch.mean(val_loss), logger=True, on_epoch=True)
        self.log('Validation/avg_val_amp_loss', torch.mean(amp_loss), logger=True, on_epoch=True)
        self.log('Validation/avg_val_phi_loss', torch.mean(phi_loss), logger=True, on_epoch=True)
    
    def load_test_model(self, path):
        self.sres = SResTransformerTrain(self.hparams.d_model,
                                            coords=self.coords,
                                            flatten_order=self.dst_flatten_order,
                                            # attention_type='full',
                                            n_layers=self.hparams.n_layers,
                                            n_heads=self.hparams.n_heads,
                                            d_query=self.hparams.d_query,
                                            dropout=self.hparams.dropout,
                                            attention_dropout=self.hparams.attention_dropout)
        if len(path) > 0:
            weights = torch.load(path)
            sd = {}
            for k in weights['state_dict'].keys():
                if k[:5] == 'sres.':
                    sd[k[5:]] = weights['state_dict'][k]
            self.sres.load_state_dict(sd)
        self.sres.to('cuda')

    def predict_and_get_lowres_pred_gt(self, fc, mag_min, mag_max):
        x_fc = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        pred = self.sres.forward_i(x_fc,self.input_seq_length)
        lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc, pred, mag_min, mag_max)

    def get_lowres_pred_gt(self, fc, pred, mag_min, mag_max):
        pred_img = self.convert2img(pred, mag_min=mag_min, mag_max=mag_max)
        lowres = torch.zeros_like(pred)
        lowres += fc.min()
        lowres[:, :self.input_seq_length] = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        lowres_img = self.convert2img(fc=lowres, mag_min=mag_min, mag_max=mag_max)
        gt_img = self.convert2img(fc=fc[:, self.dst_flatten_order], mag_min=mag_min, mag_max=mag_max)
        

        lowres_img = denormalize(lowres_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        gt_img = denormalize(gt_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)

        return lowres_img, pred_img, gt_img
    
    def on_test_epoch_end(self):
        
        lowres_psnrs = torch.stack(self.test_outputs[0])
        pred_psnrs = torch.stack(self.test_outputs[1])
        self.log('Test/Input Mean PSNR', torch.mean(lowres_psnrs), logger=True,on_epoch=True)
        self.log('Test/Input SEM PSNR', torch.std(lowres_psnrs / np.sqrt(len(lowres_psnrs))),logger=True,on_epoch=True)
        self.log('Test/Prediction Mean PSNR', torch.mean(pred_psnrs), logger=True,on_epoch=True)
        self.log('Test/Prediction SEM PSNR', torch.std(pred_psnrs / np.sqrt(len(pred_psnrs))),logger=True,on_epoch=True)

    def convert2img(self, fc, mag_min, mag_max):
        dft = convert2DFT(x=fc, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=self.dst_flatten_order,
                          img_shape=self.hparams.img_shape)
        return torch.fft.irfftn(dft, s = 2 * (self.hparams.img_shape,), dim=[1,2])

    def test_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        lowres_img,pred_img,gt_img = self.get_lowres_pred_gt(fc, mag_min, mag_max)

        # lowres_psnr = PSNR(gt_img,lowres_img)
        lowres_psnr = [PSNR(gt_img[i], lowres_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                    range(gt_img.shape[0])]
        # pred_psnr = PSNR(gt_img,pred_img)
        pred_psnr = [PSNR(gt_img[i], pred_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                    range(gt_img.shape[0])]
        self.test_outputs = [lowres_psnr, pred_psnr,pred_img, lowres_img, gt_img]
        return (lowres_psnr, pred_psnr)
    
    def on_test_epoch_end(self):
        lowres_psnrs = self.test_outputs[0]
        pred_psnrs = self.test_outputs[1]
        self.log('Input Mean PSNR', torch.mean(lowres_psnrs), logger=True)
        self.log('Input SEM PSNR', torch.std(lowres_psnrs / np.sqrt(len(lowres_psnrs))),
                 logger=True)
        self.log('Prediction Mean PSNR', torch.mean(pred_psnrs), logger=True)
        self.log('Prediction SEM PSNR', torch.std(pred_psnrs / np.sqrt(len(pred_psnrs))),
                 logger=True)
