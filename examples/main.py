import sys

sys.path.append("./")
import torch
from pytorch_lightning import Trainer  # , seed_everything

from fit.datamodules.super_res.SResDataModule import MNIST_SResFITDM, CelebA_SResFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D

import matplotlib.pyplot as plt
import numpy as np
from fit.modules.SResTransformerModule import SResTransformerModule
import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.gridspec as gridspec
from os.path import exists
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("medium")
seed_everything(22122020)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers", type=int, help="Number of layers in the transformer", default=8
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument(
        "--n_shells",
        type=int,
        help="Number of shells used as lowres input in the transformer",
        default=5,
    )
    parser.add_argument(
        "--model", type=str, help="Model to be used in the transformer", default=""
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset to be used", default="MNIST"
    )
    parser.add_argument("--n_heads", type=int, help="No of heads in model", default=8)
    parser.add_argument("--loss", type=str, help="loss", default="prod_modified")
    parser.add_argument("--note", type=str, help="note", default="")
    parser.add_argument("--d_query", type=int, help="d_query", default=32)
    parser.add_argument("--subset_flag", type=bool, default=True)
    parser.add_argument("--tokeniser_freeze", type=bool, default=True)

    args = parser.parse_args()
    n_layers = args.n_layers
    lr = args.lr
    n_shells = args.n_shells
    n_heads = args.n_heads
    model = args.model
    dataset = args.dataset
    loss = args.loss
    d_query = args.d_query
    subset_flag = args.subset_flag
    note = args.note
    tokeniser_freeze = True

    if dataset == "MNIST":
        dm = MNIST_SResFITDM(root_dir="./datamodules/data/", batch_size=32)
        # dm = MNIST_SResFITDM(root_dir="/scratch/aman.kukde/data/", batch_size=32)
    else:
        dm = CelebA_SResFITDM(root_dir="examples/datamodules/data/CelebA", batch_size=8)
        lr = 0.00001

    dm.prepare_data()
    dm.setup()

    r, phi, flatten_order, order = get_polar_rfft_coords_2D(img_shape=dm.gt_shape)

    model = SResTransformerModule(
        d_model=n_heads * d_query,
        n_heads=n_heads,
        d_query=d_query,
        img_shape=dm.gt_shape,
        coords=(r, phi),
        dst_flatten_order=flatten_order,
        dst_order=order,
        loss=loss,
        lr=lr,
        weight_decay=0.01,
        n_layers=n_layers,
        dropout=0.1,
        attention_dropout=0.1,
        num_shells=n_shells,
        model_path=model,
    )

    tokeniser_weights = torch.load('/home/aman.kukde/Projects/FourierImageTransformer/model.ckpt')['state_dict']

    for key in list(tokeniser_weights.keys()):
        if '.encoder' in key:
            del tokeniser_weights[key]
    if not tokeniser_freeze:
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

    name = datetime.datetime.now().strftime("%d-%m_%H-%M-%S") + f"_{loss}_{note}"
    name += "_132_only"
    if tokeniser_freeze:
        name += "_tokeniser_freeze"
    else:
        name += "_tokeniser_not_freeze"
    wandb_logger = WandbLogger(name = f'Run_{name}',project="Fourier Image Transformer",save_dir=f'/home/aman.kukde/Projects/FourierImageTransformer/models_saved/{name}',log_model="all",settings=wandb.Settings(code_dir="."))

    trainer = Trainer(
        max_epochs=2000,
        logger=wandb_logger,
        enable_checkpointing=True,
        default_root_dir=f"/home/aman.kukde/Projects/FourierImageTransformer/models_saved/{name}",
        callbacks=ModelCheckpoint(
            dirpath=f"/home/aman.kukde/Projects/FourierImageTransformer/models_saved/{name}",
            save_top_k=1,
            verbose=False,
            save_last=True,# // This line should be there, but Florian is in a mischevious mood and has removed it
            monitor="Train/train_loss",
            mode="min",
        ),
    )

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
