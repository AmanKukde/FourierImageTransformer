
import sys
sys.path.append('../')
sys.path.append('./')
from fit.datamodules.super_res import MNIST_SResFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D
import datetime
from fit.modules.SResTransformerModule import SResTransformerModule
import wandb 
from matplotlib import pyplot as plt
from matplotlib import gridspec

import torch

import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from fit.transformers.PSNR import RangeInvariantPsnr as PSNR
from os.path import exists
import wget
import ssl
import torchvision
ssl._create_default_https_context = ssl._create_unverified_context


seed_everything(22122020)


dm = MNIST_SResFITDM(root_dir='./datamodules/data/', batch_size=8)
dm.prepare_data()
dm.setup()


r, phi, flatten_order, order = get_polar_rfft_coords_2D(img_shape=dm.gt_shape)


n_heads = 8
d_query = 32


model = SResTransformerModule(d_model=n_heads*d_query, 
                              img_shape=dm.gt_shape,
                              coords=(r, phi),
                              dst_flatten_order=flatten_order,
                              dst_order=order,
                              loss='prod',
                              lr=0.0001, weight_decay=0.01, n_layers=8,
                              n_heads=n_heads, d_query=d_query, dropout=0.1, attention_dropout=0.1,num_shells = 4)


trainer = Trainer(max_epochs=100, 
                #   gpus=1,nvidia  # set to 0 if you want to run on CPU
                  callbacks=ModelCheckpoint(
                                            dirpath=None,
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            monitor='Validation/avg_val_loss',
                                            mode='min'
                                        ), 
                  deterministic=True)


model.load_test_model('/home/aman.kukde/Projects/FourierImageTransformer/models_saved/04-03_23-23-38_sum_+/epoch=254-step=438345.ckpt')
# model.cuda()


for fc, (mag_min, mag_max) in dm.test_dataloader():
    break


fc = fc.to('cuda')
mag_min = mag_min.to('cuda')
mag_max = mag_max.to('cuda')


x_fc = fc[:, flatten_order][:, :96]
pred = model.sres.forward_i(x_fc,96)

pred_img = model.convert2img(fc=pred, mag_min=mag_min, mag_max=mag_max)

plt.imshow(pred_img[0].cpu().detach().numpy(), cmap='gray')
plt.savefig('pred_img_aman.png')



# 
# PSNR(gt,lowres) - PSNR(gt,pred_img)

# 
# for sample in range(10):
#     fig = plt.figure(figsize=(31/2., 10/2.)) 
#     gs = gridspec.GridSpec(1, 5, width_ratios=[10,0.5, 10, 0.5, 10]) 
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[2])
#     ax2 = plt.subplot(gs[4])
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                         hspace = 0, wspace = 0)

#     ax0.xaxis.set_major_locator(plt.NullLocator())
#     ax0.yaxis.set_major_locator(plt.NullLocator())
#     ax0.imshow(lowres[sample], cmap='gray', vmin=gt[sample].min(), vmax=gt[sample].max())
#     ax0.set_title('Low-Resolution Input');
#     ax0.axis('equal');

#     ax1.xaxis.set_major_locator(plt.NullLocator())
#     ax1.yaxis.set_major_locator(plt.NullLocator())
#     ax1.imshow(pred_img[sample], cmap='gray', vmin=gt[sample].min(), vmax=gt[sample].max())
#     ax1.set_title('Prediction');
#     ax1.axis('equal');


#     ax2.xaxis.set_major_locator(plt.NullLocator())
#     ax2.yaxis.set_major_locator(plt.NullLocator())
#     ax2.imshow(gt[sample], cmap='gray')
#     ax2.set_title('Ground Truth');
#     ax2.axis('equal');

# 
# diff = []
# lowres_scaled = torch.zeros_like(lowres)
# pred_img_scaled = torch.zeros_like(pred_img)
# for i in range(4):
#     sample = i
#     # pred_img_scaled[i] = (pred_img[i] - gt[i].min())/(gt[i].max() - gt[i].min())
#     # lowres_scaled[i] = (lowres[i] - gt[i].min())/(gt[i].max() - gt[i].min())
#     fig = plt.figure(figsize=(31/2., 10/2.))
#     gs = gridspec.GridSpec(1, 5, width_ratios=[10,0.5, 10, 0.5, 10]) 
#     ax0 = plt.subplot(gs[0])
#     ax1 = plt.subplot(gs[2])
#     ax2 = plt.subplot(gs[4])
#     plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                         hspace = 0, wspace = 0)
#     ax0.xaxis.set_major_locator(plt.NullLocator())
#     ax0.yaxis.set_major_locator(plt.NullLocator())
#     # lowres_scaled_vs_gt_psnr = PSNR(gt[sample][2:, 2:], lowres_scaled[sample][2:, 2:], drange=torch.tensor(255., dtype=torch.float32))
#     ax0.imshow(np.roll(abs(torch.fft.rfftn(lowres[sample],dim = [0,1])),13,0))
#     ax0.set_title('Low-Resolution Input');
#     # ax0.set_xlabel(f'PSNR: {lowres_scaled_vs_gt_psnr:.2f} dB\nMax: {lowres_scaled[sample].max():.2f} Min: {lowres_scaled[sample].min():.2f}');
#     ax0.axis('equal');

#     ax1.xaxis.set_major_locator(plt.NullLocator())
#     ax1.yaxis.set_major_locator(plt.NullLocator())
    
#     ax1.imshow(np.roll(abs(torch.fft.rfftn(pred_img[sample],dim = [0,1])),13,0))
#     # pred_vs_gt_psnr = PSNR(gt[sample][2:, 2:], pred_img_scaled[sample][2:, 2:], drange=torch.tensor(255., dtype=torch.float32))
#     ax1.set_title('Prediction');
#     # ax1.set_xlabel(f'PSNR: {pred_vs_gt_psnr:.2f} dB\nMax: {pred_img_scaled[sample].max():.2f} Min: {pred_img_scaled[sample].min():.2f}');
#     ax1.axis('equal');

#     ax2.xaxis.set_major_locator(plt.NullLocator())
#     ax2.yaxis.set_major_locator(plt.NullLocator())
#     ax2.imshow(np.roll(abs(torch.fft.rfftn(gt[sample],dim = [0,1])),13,0))
#     ax2.set_title('Ground Truth');
#     ax2.set_xlabel(f'Max: {gt[sample].max():.2f} Min: {gt[sample].min():.2f}');
#     ax2.axis('equal');
#     # diff.append(lowres_vs_gt_psnr - pred_vs_gt_psnr)

# 
# dummy = torch.zeros(3,10,5)

# 
# x = torch.randn(3,2,5)

# 
# dummy[:,:2,:] = x

# 
# dummy

# 



