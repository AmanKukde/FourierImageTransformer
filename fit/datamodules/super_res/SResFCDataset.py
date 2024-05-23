import sys
sys.path.append('./')
import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset
import math
from fit.utils.utils import normalize_FC, log_amplitudes
from torch.nn.functional import avg_pool3d, interpolate

# class SResFourierCoefficientDataset(Dataset):
#     def __init__(self, ds, amp_min, amp_max):
#         self.ds = ds
#         if amp_min == None and amp_max == None:
#             tmp_imgs = []
#             for i in np.random.permutation(len(self.ds))[:200]:
#                 img = self.ds[i]
#                 tmp_imgs.append(img)

#             tmp_imgs = torch.stack(tmp_imgs)
#             tmp_ffts = torch.fft.rfftn(tmp_imgs, dim=[1, 2])
#             log_amps = log_amplitudes(tmp_ffts.abs())
#             self.amp_min = log_amps.min()
#             # self.amp_min = torch.tensor([0])
#             self.amp_max = log_amps.max()
#             # self.amp_max = torch.tensor([0])
       
#         else:
#             self.amp_min = amp_min
           
#             self.amp_max = amp_max


#     def __getitem__(self, item):
#         img = self.ds[item]
#         img_fft = torch.fft.rfftn(img, dim=[0, 1])
#         img_amp, img_phi = normalize_FC(img_fft, amp_min=self.amp_min, amp_max=self.amp_max)

#         fc = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1).to('cuda')
      
#         return fc, (self.amp_min.unsqueeze(-1), self.amp_max.unsqueeze(-1))

#     def __len__(self):
#         return len(self.ds)



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

        
        fc = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1).to('cuda')
        return fc, (amp_min.unsqueeze(-1), amp_max.unsqueeze(-1))
    
    def __len__(self):
        return len(self.ds)
