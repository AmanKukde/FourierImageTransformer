import sys
sys.path.append('./')
import torch
from pytorch_lightning import Trainer#, seed_everything

from fit.datamodules.super_res.SResDataModule import MNIST_SResFITDM, CelebA_SResFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D

import matplotlib.pyplot as plt
import numpy as np
from fit.modules.SResTransformerModule import SResTransformerModule
import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import matplotlib.gridspec as gridspec
from os.path import exists
import wget
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.set_float32_matmul_precision('medium')
seed_everything(22122020)

dm = CelebA_SResFITDM(root_dir='examples/datamodules/data/CelebA', batch_size = 8)
dm.prepare_data()
dm.setup()

dm = MNIST_SResFITDM(root_dir='./datamodules/data/', batch_size=32)
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
                              loss='sum',
                              lr=0.00001, weight_decay=0.01, n_layers=8,
                              n_heads=n_heads, d_query=d_query, dropout=0.1, attention_dropout=0.1,num_shells =6)


# Train your own model.
name = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")
wandb_logger = WandbLogger(name = f'Run_{name}',project="MNIST",save_dir=f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}',log_model="all")
trainer = Trainer(precision = 16,max_epochs=100,logger=wandb_logger,
                  enable_checkpointing=True,default_root_dir = f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}', 
                                            callbacks=ModelCheckpoint(
                                            dirpath=f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}',
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            monitor='Train/avg_val_loss',
                                            # mode='min'),limit_train_batches= 0.1,fast_dev_run=True)
                                            mode='min'))#,limit_train_batches= 0.1,fast_dev_run=True)

trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)#ckpt_path = '/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/examples/models/sres_1/epoch=258-step=445221.ckpt')
# trainer.test(model, datamodule=dm)