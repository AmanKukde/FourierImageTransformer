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

# dm = CelebA_SResFITDM(root_dir='examples/datamodules/data/CelebA', batch_size = 8)
# dm.prepare_data()
# dm.setup()

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
                              lr=0.0001, weight_decay=0.01, n_layers=8,
                              n_heads=n_heads, d_query=d_query, dropout=0.1, attention_dropout=0.1,num_shells =5)


# Train your own model.
name = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")
wandb_logger = WandbLogger(name = f'Run_{name}',project="MNIST",save_dir=f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}',log_model="all")
trainer = Trainer(max_epochs=2,logger=wandb_logger,
                  enable_checkpointing=True,default_root_dir = f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}', 
                                            callbacks=ModelCheckpoint(
                                            dirpath=f'/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/{name}',
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            monitor='Validation/avg_val_loss',
                                            # mode='min'),limit_train_batches= 0.1,fast_dev_run=True)
                                            mode='min'))#,limit_train_batches= 0.1,fast_dev_run=True)

trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
trainer.test(model, datamodule=dm)

# model.load_test_model('/home/aman.kukde/Projects/Super_Resolution_Task/Original_FIT/FourierImageTransformer/saved_models/12-02_16-12/last.ckpt')
# model.cpu();


# num_rings = 5

# x, y = np.meshgrid(range(model.dft_shape[1]), range(-model.dft_shape[0] // 2 + 1, model.dft_shape[0] // 2 + 1))
# radii = np.sqrt(x ** 2 + y ** 2, dtype=np.float32)
# selected_rings = np.round(radii) < num_rings

# model.input_seq_length = np.sum(selected_rings)
# plt.imshow(selected_rings)
# plt.title('Prefix');


# for fc, (mag_min, mag_max) in dm.test_dataloader():
#     break


# lowres, pred_img, gt = model.get_lowres_pred_gt(fc, mag_min, mag_max)


# sample = 30
# fig = plt.figure(figsize=(31/2., 10/2.)) 
# gs = gridspec.GridSpec(1, 5, width_ratios=[10,0.5, 10, 0.5, 10]) 
# ax0 = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[2])
# ax2 = plt.subplot(gs[4])
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#                     hspace = 0, wspace = 0)

# ax0.xaxis.set_major_locator(plt.NullLocator())
# ax0.yaxis.set_major_locator(plt.NullLocator())
# ax0.imshow(lowres[sample], cmap='gray', vmin=gt[sample].min(), vmax=gt[sample].max())
# ax0.set_title('Low-Resolution Input');
# ax0.axis('equal');

# ax1.xaxis.set_major_locator(plt.NullLocator())
# ax1.yaxis.set_major_locator(plt.NullLocator())
# ax1.imshow(pred_img[sample], cmap='gray', vmin=gt[sample].min(), vmax=gt[sample].max())
# ax1.set_title('Prediction');
# ax1.axis('equal');


# ax2.xaxis.set_major_locator(plt.NullLocator())
# ax2.yaxis.set_major_locator(plt.NullLocator())
# ax2.imshow(gt[sample], cmap='gray')
# ax2.set_title('Ground Truth');
# ax2.axis('equal');

# import matplotlib.pyplot as plt;plt.close();
# plt.imshow(np.roll(abs(rfft).T,13,1))
# x, y = np.meshgrid(range(28), range(-14, 14))
# radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32)/2., 13, 0)
# plt.plot(radii);plt.savefig('temp_2.png')