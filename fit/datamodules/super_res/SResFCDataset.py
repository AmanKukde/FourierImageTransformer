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
    def __init__(self, ds, amp_min, amp_max):
        self.ds = ds
        if amp_min == None and amp_max == None:
            tmp_imgs = []
            for i in np.random.permutation(len(self.ds))[:200]:
                img = self.ds[i]
                tmp_imgs.append(img)

            tmp_imgs = torch.stack(tmp_imgs)
            tmp_ffts = torch.fft.rfftn(tmp_imgs, dim=[1, 2])
            log_amps = log_amplitudes(tmp_ffts.abs())
            self.amp_min = log_amps.min()
            # self.amp_min = torch.tensor([0])
            self.amp_max = log_amps.max()
            # self.amp_max = torch.tensor([0])
       
        else:
            self.amp_min = amp_min
           
            self.amp_max = amp_max


    def __getitem__(self, item):
        img = self.ds[item]
        img_fft = torch.fft.rfftn(img, dim=[0, 1])
        img_amp, img_phi = normalize_FC(img_fft, amp_min=self.amp_min, amp_max=self.amp_max)

        fc = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1).to('cuda')
        
        # for r in self.tokens_per_radius:
        #     selected_ring = (radii == r)
        #     tokens_per_radius.append(radii[selected_ring].shape[-1])
        #     print(np.where(selected_ring == True))
        #     selected_ring = selected_ring.flatten()
        #     print(np.where(selected_ring == True))
        #     input_ =  fc[selected_ring]

        # # x, y = np.meshgrid(range(img_fft.shape[1]), range(-img_fft.shape[0] // 2 + 1, img_fft.shape[0] // 2 + 1))
        # # radii = np.round(np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), img_fft.shape[0] // 2 + 1, 0))
        # # tokens_per_radius = []
        # # R = img_fft.shape[0] // 2 + 1 #Radius of fft
        # # E = int((img_fft.shape[0] // 2 + 1)*np.pi) #intrapolation size
        # # Max_Number_of_Rings = math.floor((img_fft.shape[0] // 2 + 1) * 1.414) #Maximum radius
        # # Intrapolated_FFT = torch.zeros(size = (fc.shape[0],2,Max_Number_of_Rings,E)) #Collect the extrpolated tokens. (Batch, 2, Rings, Extrapolation)
        
        # # for r in range(0,Max_Number_of_Rings):
        # #     selected_ring = (radii == r)
        # #     tokens_per_radius.append(radii[selected_ring].shape[-1])
        # #     print(np.where(selected_ring == True))
        # #     selected_ring = selected_ring.flatten()
        # #     print(np.where(selected_ring == True))
        # #     input_ =  fc[selected_ring]
        #     # Intrapolated_FFT[:,r,:] = interpolate(input_, E)
        # Intrapolated_FFT = Intrapolated_FFT.permute(0,2,3,1)
        # CF = int(Intrapolated_FFT.shape[2]*0.2)

        # #Convolutional Layer
        # # Conv_Layer = torch.nn.Conv3d(L,L,(CF,1,1), stride=(CF,1,1))
        # # x = Conv_Layer(Intrapolated.unsqueeze(-1)).view(fc.shape[0],-1,2).to('cuda') #Batch, Sectors as Tokens, 2

        # #AvgPoolLayer 
        # x = avg_pool3d(Intrapolated_FFT,(1,CF,1),(1,CF,1)).to('cuda').view(fc.shape[0],-1,2)
        return fc, (self.amp_min.unsqueeze(-1), self.amp_max.unsqueeze(-1))

        # return fc , x, (self.amp_min.unsqueeze(-1), self.amp_max.unsqueeze(-1))

    def __len__(self):
        return len(self.ds)
