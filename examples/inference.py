import sys
sys.path.append("./")

import torch
import wandb
import ssl
import datetime
import matplotlib
from tqdm import tqdm

from fit.utils.tomo_utils import get_polar_rfft_coords_2D
from fit.modules.SResTransformerModule import SResTransformerModule
from fit.datamodules.super_res.SResDataModule import MNIST_SResFITDM, CelebA_SResFITDM

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

from utils.PSNR import RangeInvariantPsnr as PSNR

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("medium")
seed_everything(22122020)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--causal_mask", action="store_true", help="Use causal mask", default=True)
    parser.add_argument("--d_query", type=int, help="d_query", default=32)
    parser.add_argument("--model_type", type=str, help="Model to be used in the transformer (torch or fast)", default="fast")
    parser.add_argument("--n_layers", type=int, help="Number of layers in the transformer", default=8)
    parser.add_argument("--n_heads", type=int, help="No of heads in the transformer", default=8)
    parser.add_argument("--n_shells",type=int,help="Number of shells used as lowres-input in the transformer",default=5)
    parser.add_argument("--models_save_path", type=str, default="/home/aman.kukde/Projects/FourierImageTransformer/models/")
    parser.add_argument("--model_name", type=str, default= 'Fast_prod__L_8_H_8_s_5_subset_False_27-03_16-58-36/epoch=222-step=383337.ckpt')

    args = parser.parse_args()

    n_layers = args.n_layers
    n_shells = args.n_shells
    n_heads = args.n_heads
    model_type = args.model_type
    models_save_path = args.models_save_path
    model_name = args.model_name
    causal_mask = args.causal_mask
    d_query = args.d_query

    dm = MNIST_SResFITDM(root_dir="./datamodules/data/", batch_size=32)
    dm.prepare_data()
    dm.setup()

    r, phi, flatten_order, order = get_polar_rfft_coords_2D(img_shape=dm.gt_shape)

    model = SResTransformerModule(
        n_heads=n_heads,
        d_query=d_query,
        img_shape=dm.gt_shape,
        coords=(r, phi),
        model_type=model_type,
        dst_flatten_order=flatten_order,
        dst_order=order,
        weight_decay=0.01,
        n_layers=n_layers,
        num_shells=n_shells,
    )
    print(f"\n\n\n\n{model}\n\n\n\n")
    weights = torch.load(models_save_path + model_name)['state_dict']
    model.load_state_dict(weights)
    print(f"Model loaded successfully {model_name}")

    trainer = Trainer(max_epochs=100, 
                    callbacks=ModelCheckpoint(
                                                dirpath=None,
                                                save_top_k=1,
                                                verbose=False,
                                                save_last=True,
                                                monitor='Validation/avg_val_loss',
                                                mode='min'
                                            ), 
                    deterministic=True)

    model.cuda()

def make_figs(lowres_psnr, pred_psnr):
    fig = plt.figure()
    sns.histplot(lowres_psnr.cpu().detach(), kde=True, color='blue',legend =True,label = "lowres")
    sns.histplot(pred_psnr.cpu().detach(), kde=True, color='red', legend= True, label = "pred")
    fig.legend()
    plt.savefig('./inference_results/psnr_hist.png')
    plt.close()

    fig = plt.figure()
    sns.histplot(pred_psnr.cpu().detach() - lowres_psnr.cpu().detach(), kde=True, color='green', legend= True, label = "diff")
    fig.legend()
    plt.savefig('./inference_results/psnr_diff.png')
    plt.close()

    fig = plt.figure()
    sns.boxplot(lowres_psnr.cpu().detach(), color='blue',legend = True)
    fig.legend(["lowres"])
    sns.boxplot(pred_psnr.cpu().detach(), color='red',legend = True)
    plt.savefig('./inference_results/psnr_box.png')
    plt.close()
    return None

lowres_psnr = []
pred_psnr = []


for fc, (mag_min, mag_max) in tqdm(dm.test_dataloader()):
    fc = fc.cuda()
    mag_min = mag_min.cuda()
    mag_max = mag_max.cuda()
    x_fc = fc[:, flatten_order][:, :model.input_seq_length].cuda()
    pred = model.sres.forward_inference(x_fc)

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
torch.save(lowres_psnr, './inference_results/lowres.pt')
torch.save(pred_psnr,'./inference_results/pred.pt')










