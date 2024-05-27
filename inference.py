import sys

from sklearn import model_selection
sys.path.append('../')
from fit.datamodules.super_res import MNIST_SResFITDM, CelebA_SResFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D
from fit.modules.SResTransformerModule import SResTransformerModule

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from tqdm import tqdm
from fit.utils.PSNR import RangeInvariantPsnr as PSNR
import datetime
import torch
import numpy as np
from pytorch_lightning import seed_everything
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt


class Inference:

    def __init__(self, ckpt_path_list=[], save_folder = '../'):
        self.ckpt_path_list = ckpt_path_list
        self.save_folder = save_folder

    def run_inference(self):
        for path in self.ckpt_path_list:
            model_type, model, loss, dataset, dm, flatten_order,model_created = self._load_model_from_ckpt_path(
                ckpt_path=path)
            save_loc = self.save_folder + f"/{model_type}_{dataset}/{loss}/{model_created}/{datetime.datetime.now().strftime('%m_%d_%H_%M')}"
            Path(save_loc).mkdir(parents=True, exist_ok=True)
                
            self.test_model_save_pred_psnr(
                model_type,
                model,
                dm,
                flatten_order,
                save_location=save_loc)

    def _load_model_from_ckpt_path(self, ckpt_path):
        #/home/aman.kukde/Projects/FourierImageTransformer/models/MNIST/mamba/sum/Mamba_MNIST_wp_1000_sum_L_8_H_8_s_10_subset_False_23-05_11-56-51/epoch=80-step=69660.ckpt
        ckpt_path = ckpt_path
        dataset = ckpt_path.split('/')[-5]
        model_type = ckpt_path.split('/')[-4]
        loss = ckpt_path.split('/')[-3]
        model_created = (ckpt_path.split('/')[-2]).split('_')[-1]
        seed_everything(22122020)

        if dataset == "MNIST":
            dm = MNIST_SResFITDM(root_dir="./datamodules/data/",
                                 batch_size=64,
                                 subset_flag=False)
        if dataset == "CelebA":
            dm = CelebA_SResFITDM(root_dir="./datamodules/data/",
                                  batch_size=8,
                                  subset_flag=False)
        dm.prepare_data()
        dm.setup()

        r, phi, flatten_order, order = get_polar_rfft_coords_2D(
            img_shape=dm.gt_shape)
        n_heads = 8
        d_query = 32
        model = SResTransformerModule(img_shape=dm.gt_shape,
                                      coords=(r, phi),
                                      dst_flatten_order=flatten_order,
                                      dst_order=order,
                                      lr=0.0001,
                                      weight_decay=0.01,
                                      n_layers=8,
                                      n_heads=n_heads,
                                      d_query=d_query,
                                      num_shells=5,
                                      model_type=model_type)

        weights = torch.load(ckpt_path)['state_dict']
        model.load_state_dict(weights, strict=True)
        model.cuda()
        model.eval()
        print('Model Loaded')
        return model_type, model, loss, dataset, dm, flatten_order,model_created

    def test_model_save_pred_psnr(self, model_type, model, dm, flatten_order,
                                  save_location):
        lowres_psnr = []
        pred_psnr = []
        for fc, (mag_min, mag_max) in tqdm(dm.test_dataloader()):
            fc = fc.cuda()
            mag_min = mag_min.cuda()
            mag_max = mag_max.cuda()
            x_fc = fc[:, flatten_order]
            pred = model.sres.forward_i(x_fc)

            pred_img = model.convert2img(fc=pred,
                                         mag_min=mag_min,
                                         mag_max=mag_max)

            lowres = torch.zeros_like(pred)
            lowres += fc.min()
            lowres[:, :model.
                   input_seq_length] = fc[:, model.
                                          dst_flatten_order][:, :model.
                                                             input_seq_length]
            lowres_img = model.convert2img(fc=lowres,
                                           mag_min=mag_min,
                                           mag_max=mag_max)
            gt_img = model.convert2img(fc=fc[:, model.dst_flatten_order],
                                       mag_min=mag_min,
                                       mag_max=mag_max)

            lowres_psnr.append(PSNR(gt_img, lowres_img))
            pred_psnr.append(PSNR(gt_img, pred_img))

        lowres_psnr = torch.concat(lowres_psnr)
        pred_psnr = torch.concat(pred_psnr)
        torch.save(lowres_psnr, f'{save_location}/{model_type}_lowres.pt')
        torch.save(pred_psnr, f'{save_location}/{model_type}_pred.pt')
        return lowres_psnr, pred_psnr

    def make_figs_individual(self, input_dict, save_folder, show=False):
        font = {'family': 'serif', 'weight': 'normal', 'size': 16}
        matplotlib.rc('font', **font)
        #Indivitual models
        for id in input_dict.keys():
            lowres_psnr, pred_psnr = input_dict[id]
            fig = plt.figure(figsize=(12, 9))
            sns.histplot(lowres_psnr.cpu().detach(),
                         kde=True,
                         color='blue',
                         legend=True,
                         label="lowres")
            sns.histplot(pred_psnr.cpu().detach(),
                         kde=True,
                         color='red',
                         legend=True,
                         label="pred")
            fig.legend()
            plt.savefig(f'{save_folder}/{id}_psnr_hist.png')
            if show: plt.show()
            plt.close()

            fig = plt.figure(figsize=(12, 9))
            sns.histplot(pred_psnr.cpu().detach() - lowres_psnr.cpu().detach(),
                         kde=True,
                         color='green',
                         legend=True,
                         label="diff")
            fig.legend()
            plt.savefig(f'{save_folder}/{id}_psnr_diff.png')
            if show: plt.show()
            plt.close()

            plt.figure(figsize=(12, 9))
            plt.boxplot(
                [lowres_psnr, pred_psnr, pred_psnr - lowres_psnr],
                widths=[0.9] * 3,
                labels=['lowres_psnr', 'pred_psnr', 'diff (pred - lowres)'])
            plt.savefig(f'{save_folder}/{id}_psnr_box_LvsPvsD.png')
            if show: plt.show()
            plt.close()

            plt.figure(figsize=(12, 9))
            plt.boxplot([lowres_psnr, pred_psnr],
                        widths=[0.9] * 2,
                        labels=['lowres_psnr', 'pred_psnr'])
            plt.savefig(f'{save_folder}/{id}_psnr_box_LvsP.png')
            if show: plt.show()
            plt.close()

            plt.figure(figsize=(12, 9))
            plt.boxplot([pred_psnr - lowres_psnr],
                        widths=[0.9],
                        labels=['diff (pred - lowres)'])
            plt.savefig(f'{save_folder}/{id}_psnr_box_diff.png')
            if show: plt.show()
            plt.close()

            diff = np.sort(pred_psnr - lowres_psnr)
            p = np.arange(0, 101, 1)
            xt = np.arange(0, 105, 5)
            perc = np.percentile(diff, q=p)
            plt.figure(figsize=(10, 10))
            plt.plot(diff, label='PSNR Difference Prediction - Lowres')
            plt.plot((len(diff) + 1) * p / 100.,
                     perc,
                     'ro',
                     label='+1 Percentile of PSNR Difference Distribution')
            plt.xticks((len(diff) - 1) * xt / 100., map(str, xt))
            plt.legend()
            plt.grid()
            plt.savefig(f'{save_folder}/{id}_psnr_diff_percentile.png')
            if show: plt.show()
            plt.close()

    def make_figs_comparison(self, input_dict, save_folder, show=False):
        font = {'family': 'serif', 'weight': 'normal', 'size': 16}
        matplotlib.rc('font', **font)
        #Comparison models
        fig = plt.figure(figsize=(12, 9))
        for id in input_dict.keys():
            lowres_psnr, pred_psnr = input_dict[id]
            sns.histplot(lowres_psnr.cpu().detach(),
                         kde=True,
                         legend=True,
                         label=f'{id}_lowres')
            sns.histplot(pred_psnr.cpu().detach(),
                         kde=True,
                         legend=True,
                         label=f'{id}_pred')
        fig.legend()
        plt.savefig(f'{save_folder}/Comparison_psnr_hist.png')
        if show: plt.show()
        plt.close()

        fig = plt.figure(figsize=(12, 9))
        for id in input_dict.keys():
            lowres_psnr, pred_psnr = input_dict[id]
            sns.histplot(pred_psnr.cpu().detach() - lowres_psnr.cpu().detach(),
                         kde=True,
                         legend=True,
                         label=f"{id}_diff")
        fig.legend()
        plt.savefig(f'{save_folder}/Comparison_psnr_diff.png')
        if show: plt.show()
        plt.close()

        plt.figure(figsize=(12, 9))
        diff = {}
        for id in input_dict.keys():
            lowres_psnr, pred_psnr = input_dict[id]
            diff[id] = (pred_psnr - lowres_psnr).cpu().detach().numpy()

        plt.boxplot([diff[key] for key in diff.keys()],
                    widths=[0.9] * len(diff.keys()),
                    labels=[f"{key}_psnr" for key in diff.keys()])
        plt.savefig(f'{save_folder}/Comparison_psnr_box.png')
        if show: plt.show()
        plt.close()

        diff = np.sort(pred_psnr - lowres_psnr)
        p = np.arange(0, 101, 1)
        xt = np.arange(0, 105, 5)
        perc = np.percentile(diff, q=p)
        plt.figure(figsize=(10, 10))
        for id in input_dict.keys():
            lowres_psnr, pred_psnr = input_dict[id]
            plt.plot(diff, label='PSNR Difference Prediction - Lowres')
            plt.plot(
                (len(diff) + 1) * p / 100.,
                perc,
                'o',
                label=f'{id}_+1 Percentile of PSNR Difference Distribution')
            plt.xticks((len(diff) - 1) * xt / 100., map(str, xt))
        plt.legend()
        plt.grid()
        plt.savefig(f'{save_folder}/Comparison_psnr_diff_percentile.png')
        if show: plt.show()
        plt.close()

        return None



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="")
    parser.add_argument("--save_folder", type=str, help="Folder to save the results", default="./inference_results/")
    args = parser.parse_args()
    Inference = Inference(ckpt_path_list = [args.ckpt_path],save_folder = args.save_folder)
    Inference.run_inference()

