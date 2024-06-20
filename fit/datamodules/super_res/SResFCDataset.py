import sys
sys.path.append('./')
import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset
import math
from fit.utils.utils import normalize_FC, log_amplitudes
from torch.nn.functional import avg_pool3d, interpolate

class SResFourierCoefficientDataset(Dataset):
    def __init__(self, ds):#, amp_min, amp_max):
        self.ds = ds

    def __getitem__(self, item):
        img = self.ds[item]
        img_fft = torch.fft.rfftn(img, dim=[0, 1])
        log_amps = log_amplitudes(img_fft.abs())
        
        amp_min = log_amps.min()
        amp_max = log_amps.max()
        
        img_amp, img_phi = normalize_FC(img_fft, amp_min=amp_min, amp_max=amp_max)

        fc = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1)
        return fc, (amp_min.unsqueeze(-1), amp_max.unsqueeze(-1))
    
    def __len__(self):
        return len(self.ds)
