from os.path import join, exists
from typing import Optional, Union, List

import numpy as np
import torch
import wget
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import resize

from fit.datamodules.super_res import SResFourierCoefficientDataset
from fit.datamodules.GroundTruthDatasetFactory import GroundTruthDatasetFactory
from fit.utils.utils import normalize
from fit.utils import read_mrc


class SResFITDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size, gt_shape):
        """

        :param root_dir:
        :param batch_size:
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.gt_shape = gt_shape
        self.gt_ds = None
        self.mean = None
        self.std = None
        self.mag_min = None
        self.mag_max = None

    def setup(self, stage: Optional[str] = None):
        tmp_fcds = SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'), amp_min=None,
                                                 amp_max=None)
        self.mag_min = tmp_fcds.amp_min
        self.mag_max = tmp_fcds.amp_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'), amp_min=self.mag_min,
                                          amp_max=self.mag_max),
            batch_size=self.batch_size, num_workers=0)

    # def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='validation'), amp_min=self.mag_min,
    #                                       amp_max=self.mag_max),
    #         batch_size=self.batch_size, num_workers=0)

    # def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='test'), amp_min=self.mag_min,
    #                                       amp_max=self.mag_max),
    #         batch_size=self.batch_size)


class MNIST_SResFITDM(SResFITDataModule):

    def __init__(self, root_dir, batch_size, subset_flag = False):
        """
        Uses the MNIST[1] dataset via the PyTorch API.

        :param root_dir:
        :param batch_size:

        References:
            [1] Yann LeCun and Corinna Cortes.
            MNIST handwritten digit database. 2010.
        """
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=27)

    def prepare_data(self, *args, **kwargs):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data.type(torch.float32)
        mnist_train_val = MNIST(self.root_dir, train=True, download=True).data.type(torch.float32)
        np.random.seed(1612)

        if self.subset_flag:
            print("Using subset of MNIST dataset")
            mnist_train = mnist_train_val[114, 1:, 1:]
            mnist_train = torch.tile(mnist_train, (self.batch_size*100, 1, 1))
            mnist_val = mnist_train.clone()
        else :
            print("Using Full MNIST dataset")
            perm = np.random.permutation(mnist_train_val.shape[0])
            mnist_train = mnist_train_val[perm[:55000],1:,1:]
            mnist_val = mnist_train_val[perm[55000:], 1:, 1:]

        mnist_test = mnist_test[:, 1:, 1:]
        self.mean = mnist_train.mean()
        self.std = mnist_train.std()

        mnist_train = normalize(mnist_train, self.mean, self.std)
        mnist_val = normalize(mnist_val, self.mean, self.std)
        mnist_test = normalize(mnist_test, self.mean, self.std)

        self.gt_ds = GroundTruthDatasetFactory(mnist_train, mnist_val, mnist_test)
class MNIST_SResFITDM_Large(SResFITDataModule):

    def __init__(self, root_dir, batch_size, subset_flag = False):
        """
        Uses the MNIST[1] dataset via the PyTorch API.

        :param root_dir:
        :param batch_size:

        References:
            [1] Yann LeCun and Corinna Cortes.
            MNIST handwritten digit database. 2010.
        """
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=129)

    def prepare_data(self, *args, **kwargs):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data.type(torch.float32)
        mnist_train_val = MNIST(self.root_dir, train=True, download=True).data.type(torch.float32)
        mnist_train_val = resize(mnist_train_val, (130, 130))
        mnist_test = resize(mnist_test, (130, 130))
        np.random.seed(1612)

        if self.subset_flag:
            print("Using subset of MNIST dataset")
            mnist_train = mnist_train_val[114, 1:, 1:]
            mnist_train = torch.tile(mnist_train, (self.batch_size*100, 1, 1))
            mnist_val = mnist_train.clone()
        else :
            print("Using Full MNIST dataset")
            perm = np.random.permutation(mnist_train_val.shape[0])
            mnist_train = mnist_train_val[perm[:55000],1:,1:]
            mnist_val = mnist_train_val[perm[55000:], 1:, 1:]

        mnist_test = mnist_test[:, 1:, 1:]
        self.mean = mnist_train.mean()
        self.std = mnist_train.std()

        mnist_train = normalize(mnist_train, self.mean, self.std)
        mnist_val = normalize(mnist_val, self.mean, self.std)
        mnist_test = normalize(mnist_test, self.mean, self.std)

        self.gt_ds = GroundTruthDatasetFactory(mnist_train, mnist_val, mnist_test)


class CelebA_SResFITDM(SResFITDataModule):

    def __init__(self, root_dir, batch_size, subset_flag = False):
        """
        Uses the CelebA[1] dataset.

        :param root_dir:
        :param batch_size:

        References:
            [1] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
            Deep learning face attributes in the wild.
            In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
        """
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=63)

    def prepare_data(self, *args, **kwargs):
        if not exists(join(self.root_dir, 'gt_data.npz')):
            wget.download('https://cloud.mpi-cbg.de/index.php/s/Wtuy9IqUsSpjKav/download',
                          out=join(self.root_dir, 'gt_data.npz'))

        gt_data = np.load(join(self.root_dir, 'gt_data.npz'))

        gt_train = torch.from_numpy(gt_data['gt_train'])
        gt_val = torch.from_numpy(gt_data['gt_val'])
        gt_test = torch.from_numpy(gt_data['gt_test'])
        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)
        self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)

class NotMNIST_SResFITDM(SResFITDataModule):
    
        def __init__(self, root_dir, batch_size, subset_flag = False):
            """
            Uses the NotMNIST[1] dataset.
    
            :param root_dir:
            :param batch_size:
    
            References:
                [1] Yaroslav Bulatov.
                NotMNIST dataset. 2011.
            """
            self.subset_flag = subset_flag
            self.batch_size = batch_size
            super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=27)
    
        def prepare_data(self, *args, **kwargs):
            if not exists(join(self.root_dir, 'gt_data.npz')):
                wget.download('https://cloud.mpi-cbg.de/index.php/s/Wtuy9IqUsSpjKav/download',
                            out=join(self.root_dir, 'gt_data.npz'))
    
            gt_data = np.load(join(self.root_dir, 'gt_data.npz'))
    
            gt_train = torch.from_numpy(gt_data['gt_train'])
            gt_val = torch.from_numpy(gt_data['gt_val'])
            gt_test = torch.from_numpy(gt_data['gt_test'])
            self.mean = gt_train.mean()
            self.std = gt_train.std()
    
            gt_train = normalize(gt_train, self.mean, self.std)
            gt_val = normalize(gt_val, self.mean, self.std)
            gt_test = normalize(gt_test, self.mean, self.std)
            self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)


class BioSRMicrotubules(SResFITDataModule):
    def __init__(self, root_dir, batch_size, subset_flag=False):
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=129)

    def prepare_data(self, *args, **kwargs):
        images = read_mrc.read_mrc('/group/jug/ashesh/data/BioSR/Microtubules/GT_all.mrc')[1]
        images = torch.permute(torch.from_numpy(images).type(torch.float32),(2,0,1))
        images = resize(images, (129, 129))
        np.random.seed(1612)
        gt_train = images[:45]
        gt_val = images[45:50]
        gt_test = images[50:]
        
        self.mean = images.mean()
        self.std = images.std()
        del images
        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)
        self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)

class BioSRFActin(SResFITDataModule):
    def __init__(self, root_dir, batch_size, subset_flag=False):
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=129)

    def prepare_data(self, *args, **kwargs):
        images = read_mrc.read_mrc('/group/jug/ashesh/data/BioSR/F-actin_Nonlinear/GT_all_b.mrc')[1]
        images = torch.permute(torch.from_numpy(images).type(torch.float32),(2,0,1))
        images = resize(images, (129, 129))
        np.random.seed(1612)
        gt_train = images[:45]
        gt_val = images[45:50]
        gt_test = images[50:]
        
        self.mean = images.mean()
        self.std = images.std()
        del images
        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)
        self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)


class Omniglot(SResFITDataModule):
    def __init__(self, root_dir, batch_size, subset_flag=False):
        self.subset_flag = subset_flag
        self.batch_size = batch_size
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=105)

    def prepare_data(self, *args, **kwargs):
        images  = torch.load('/home/aman.kukde/Projects/FourierImageTransformer/examples/datamodules/data/omniglot/Omniglot.pt')
        t = len(images)
        np.random.seed(1612)
        gt_train = images[:t*85//100]
        gt_val = images[t*85//100:t*90//100]
        gt_test = images[t*9//10:]
        
        self.mean = images.mean()
        self.std = images.std()
        del images
        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)
        self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)
