from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from fit.datamodules.super_res import MNIST_SResFITDM, CelebA_SResFITDM,BioSRMicrotubules,Omniglot, MNIST_SResFITDM_Large
from fit.modules.SResTransformerModule import SResTransformerModule
from fit.utils.tomo_utils import get_polar_rfft_coords_2D
from pathlib import Path
import datetime
import ssl
import wandb
import torch
import sys
sys.path.append("./")


ssl._create_default_https_context = ssl._create_unverified_context
# torch.set_float32_matmul_precision("medium")
seed_everything(22122020)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, help="Job Id of Slurm Job", default="")
    parser.add_argument("--num_nodes", type=int, help="Number of nodes", default=1)
    parser.add_argument("--d_query", type=int, help="d_query", default=32)
    parser.add_argument("--dataset", type=str,
                        help="Dataset to be used", default="MNIST")
    parser.add_argument("--loss", type=str, help="loss", default="sum")
    parser.add_argument("--lr", type=float,
                        help="Learning rate", default=0.0001)
    parser.add_argument("--model_type", type=str,
                        help="Model to be used in the transformer (torch or fast or mamba)", default="mamba")
    parser.add_argument("--n_layers", type=int,
                        help="Number of layers in the transformer", default=8)
    parser.add_argument("--n_heads", type=int,
                        help="No of heads in the transformer", default=8)
    parser.add_argument("--n_shells", type=int,
                        help="Number of shells used as lowres-input in the transformer", default=10)
    parser.add_argument("--subset_flag", action="store_true",
                        help="Use subset of the dataset")
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb for logging", default= False)
    parser.add_argument("--note", type=str, help="note", default="")
    parser.add_argument("--w_phi", type=float,
                        help="Weight for phi loss", default=1000)
    parser.add_argument("--models_save_path", type=str,
                        default="/home/aman.kukde/Projects/FourierImageTransformer/models")
    parser.add_argument("--resume_training_from_checkpoint",
                        type=str, default=None)
    parser.add_argument("--model_weights", type=str,default='')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_of_sectors", type=int, default=10)
    parser.add_argument("--semi_circle_only_flag", action="store_true", help="Use only semi-circle in the model", default=False)
    args = parser.parse_args()

    if args.dataset == "MNIST":
        dm = MNIST_SResFITDM(root_dir="./datamodules/data/",
                             batch_size=args.batch_size, subset_flag=args.subset_flag)
    if args.dataset == "MNIST_Large":
        dm = MNIST_SResFITDM_Large(root_dir="./datamodules/data/",
                             batch_size=args.batch_size, subset_flag=args.subset_flag)
    if args.dataset == "CelebA":
        dm = CelebA_SResFITDM(root_dir="./datamodules/data/",
                              batch_size=args.batch_size, subset_flag=args.subset_flag)
    if args.dataset == "BioSr":
        dm = BioSRMicrotubules(root_dir="./datamodules/data/",
                               batch_size=args.batch_size, subset_flag=args.subset_flag)
    if args.dataset == "Omniglot":
        dm = Omniglot(root_dir="./datamodules/data/",
                                batch_size=args.batch_size, subset_flag=args.subset_flag)
    
    

    dm.prepare_data()
    dm.setup()

    r, phi, flatten_order, order, fc_per_ring = get_polar_rfft_coords_2D(
    img_shape=dm.gt_shape)

    model = SResTransformerModule(
        n_heads=args.n_heads,
        d_query=args.d_query,
        img_shape=dm.gt_shape,
        coords=(r, phi),
        model_type=args.model_type,
        dst_flatten_order=flatten_order,
        dst_order=order,
        loss=args.loss,
        lr=args.lr,
        weight_decay=0.01,
        semi_circle_only_flag=args.semi_circle_only_flag,
        no_of_sectors=args.no_of_sectors,
        n_layers=args.n_layers,
        num_shells=args.n_shells,
        w_phi=args.w_phi,
        job_id = args.job_id,
        fc_per_ring = fc_per_ring
    )
    print(f"\n\n\n\n{model}\n\n\n\n")

    models_save_path = f"{args.models_save_path}/{args.dataset}/{args.model_type}/{args.loss}"
    Path(models_save_path).mkdir(parents=True, exist_ok=True)
    w = "wp_1"
    if args.w_phi != 1:
        w = f"wp_{int(args.w_phi)}"
    if args.resume_training_from_checkpoint != None :
        args.note += f"Contd_{args.resume_training_from_checkpoint.split('/')[-2].split('_')[-1]}"
        

    name = str.capitalize(args.model_type) + f"_{args.dataset}_{w}_{args.loss}_L_{args.n_layers}_H_{args.n_heads}_s_{args.n_shells}_subset_{args.subset_flag}_{args.note}" + \
        datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{name}",
            project="Mamba_FIT",
            save_dir=f"{models_save_path}/{name}",
            log_model="best",
            settings=wandb.Settings(code_dir="."),
        )
    else:
        wandb_logger = None
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        num_sanity_val_steps=0,
        max_epochs=10000,
        logger=wandb_logger,
        devices = -1,
        # precision='16-mixed',
        num_nodes = args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_checkpointing=True,
        gradient_clip_val=0.3,
        default_root_dir=f"{models_save_path}/{name}",
        callbacks=[ModelCheckpoint(
            dirpath=f"{models_save_path}/{name}",
            save_top_k=10,
            verbose=False,
            save_last=True,
            
            monitor="Train/train_mean_epoch_phi_loss",
            mode="min",
        ), lr_monitor],
    )
    if args.model_weights != '':
        model.load_state_dict(torch.load(args.model_weights)["state_dict"],strict=True)

    trainer.fit(model, datamodule=dm,ckpt_path=args.resume_training_from_checkpoint)
    # trainer.validate_loop._results.clear()
    # trainer.validate(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)
