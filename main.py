import sys
sys.path.append("./")

import torch
import wandb
import ssl
import datetime

from fit.utils.tomo_utils import get_polar_rfft_coords_2D
from fit.modules.SResTransformerModule import SResTransformerModule
from fit.datamodules.super_res.SResDataModule import MNIST_SResFITDM, CelebA_SResFITDM

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("medium")
seed_everything(22122020)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--d_query", type=int, help="d_query", default=32)
    parser.add_argument("--dataset", type=str, help="Dataset to be used", default="MNIST")
    parser.add_argument("--loss", type=str, help="loss", default="prod")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0001)
    parser.add_argument("--model_type", type=str, help="Model to be used in the transformer (torch or fast)", default="torch")
    parser.add_argument("--n_layers", type=int, help="Number of layers in the transformer", default=8)
    parser.add_argument("--n_heads", type=int, help="No of heads in the transformer", default=8)
    parser.add_argument("--n_shells",type=int,help="Number of shells used as lowres-input in the transformer",default=5)
    parser.add_argument("--subset_flag", action="store_true", help="Use subset of the dataset")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging", default=False)
    parser.add_argument("--note", type=str, help="note", default="")
    parser.add_argument("--models_save_path", type=str, default="/home/aman.kukde/Projects/FourierImageTransformer/models/")


    args = parser.parse_args()
    n_layers = args.n_layers
    lr = args.lr
    n_shells = args.n_shells
    n_heads = args.n_heads
    dataset = args.dataset
    loss = args.loss
    d_query = args.d_query
    note = args.note
    wandb_flag = args.wandb
    model_type = args.model_type
    subset_flag = args.subset_flag
    models_save_path = args.models_save_path

    dm = MNIST_SResFITDM(root_dir="./datamodules/data/", batch_size=32,subset_flag = subset_flag)
    dm.prepare_data()
    dm.setup()

    r, phi, flatten_order, order = get_polar_rfft_coords_2D(img_shape=dm.gt_shape)

    model = SResTransformerModule(
        d_model=n_heads * d_query,
        n_heads=n_heads,
        d_query=d_query,
        img_shape=dm.gt_shape,
        coords=(r, phi),
        model_type=model_type,
        dst_flatten_order=flatten_order,
        dst_order=order,
        loss=loss,
        lr=lr,
        weight_decay=0.01,
        n_layers=n_layers,
        dropout=0.1,
        attention_dropout=0.1,
        num_shells=n_shells,
    )
    print(f"\n\n\n\n{model}\n\n\n\n")
    name = str.capitalize(model_type) + f"_{loss}_{note}_L_{n_layers}_H_{n_heads}_s_{n_shells}_subset_{subset_flag}_" + datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

    if wandb_flag:
        wandb_logger = WandbLogger(
            name=f"{name}",
            project="Fourier Image Transformer",
            save_dir=f"{models_save_path}/{name}",
            log_model="best",
            settings=wandb.Settings(code_dir="."),
        )
    else: 
        wandb_logger = None

    trainer = Trainer(
        max_epochs=1000,
        logger=wandb_logger,
        enable_checkpointing=True,
        default_root_dir=f"{models_save_path}/{name}",
        callbacks=ModelCheckpoint(
            dirpath=f"{models_save_path}/{name}",
            save_top_k=1,
            verbose=False,
            save_last=True, 
            monitor="Validation/avg_val_loss",
            mode="min",
        ),
    )

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)