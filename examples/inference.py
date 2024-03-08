

import sys
sys.path.append('./')
from fit.datamodules.super_res import MNIST_SResFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D
import seaborn as sns
from fit.modules.SResTransformerModule import SResTransformerModule

from matplotlib import pyplot as plt
from matplotlib import gridspec
from fit.transformers.PSNR import RangeInvariantPsnr as PSNR
# from fit.utils.utils import PSNR
import torch

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from os.path import exists
import wget
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy


seed_everything(22122020)


dm = MNIST_SResFITDM(root_dir='./datamodules/data/', batch_size=32)
dm.prepare_data(subset_flag=False)
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
                              n_heads=n_heads, d_query=d_query, dropout=0.1, attention_dropout=0.1,num_shells = 5,model_path = '')

model.load_test_model('/home/aman.kukde/Projects/FourierImageTransformer/models_saved/model_main_07032024/epoch=943-step=944.ckpt')
# model.cuda()
tokeniser_weights = torch.load('/home/aman.kukde/Projects/FourierImageTransformer/model.ckpt')['state_dict']
for key in list(tokeniser_weights.keys()):
    if '.encoder' in key:
        del tokeniser_weights[key]

def load_partial_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            print(f'Copying {name}')
            if own_state[name].size() == param.size():
                own_state[name].copy_(param)
                own_state[name].requires_grad = False
                own_state[name].training = False
        # else:
        #     print(f'Layer {name} not found in current model')
    model.load_state_dict(tokeniser_weights, strict=False)
    return model

model = load_partial_state_dict(model, tokeniser_weights)

trainer = Trainer(max_epochs=100, 
                  #gpus=1, # set to 0 if you want to run on CPU
                  callbacks=ModelCheckpoint(
                                            dirpath=None,
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            monitor='Validation/avg_val_loss',
                                            mode='min'
                                        ), 
                  deterministic=True)

lowres_psnr = []
pred_psnr = []

def make_figs(lowres_psnr, pred_psnr):
    fig = plt.figure()
    sns.histplot(lowres_psnr.cpu().detach(), kde=True, color='blue',legend =True,label = "lowres")
    sns.histplot(pred_psnr.cpu().detach(), kde=True, color='red', legend= True, label = "pred")
    fig.legend()
    plt.savefig('psnr_hist.png')
    plt.close()

    fig = plt.figure()
    sns.histplot(pred_psnr.cpu().detach() - lowres_psnr.cpu().detach(), kde=True, color='green', legend= True, label = "diff")
    fig.legend()
    plt.savefig('psnr_diff.png')
    plt.close()

    fig = plt.figure()
    sns.boxplot(lowres_psnr.cpu().detach(), color='blue',legend = True)
    fig.legend(["lowres"])
    sns.boxplot(pred_psnr.cpu().detach(), color='red',legend = True)
    plt.savefig('psnr_box.png')
    plt.close()

for fc, (mag_min, mag_max) in dm.test_dataloader():
    fc = fc.to('cuda')
    mag_min = mag_min.to('cuda')
    mag_max = mag_max.to('cuda')
    x_fc = fc[:, flatten_order][:, :96]
    pred = model.sres.forward_i(x_fc,96)

    pred_img = model.convert2img(fc=pred, mag_min=mag_min, mag_max=mag_max)
 

    lowres = torch.zeros_like(pred)
    lowres += fc.min()
    lowres[:, :model.input_seq_length] = fc[:, model.dst_flatten_order][:, :model.input_seq_length]
    lowres_img = model.convert2img(fc=lowres, mag_min=mag_min, mag_max=mag_max)
    gt_img = model.convert2img(fc=fc[:, model.dst_flatten_order], mag_min=mag_min, mag_max=mag_max)

    lowres_psnr.append(PSNR(gt_img,lowres_img))
    pred_psnr.append(PSNR(gt_img,pred_img))
    make_figs(torch.concat(lowres_psnr), torch.concat(pred_psnr))
lowres_psnr = torch.concat(lowres_psnr)
pred_psnr = torch.concat(pred_psnr)
print(lowres_psnr.shape)










