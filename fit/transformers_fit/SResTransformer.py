# This file contains the implementation of the SResTransformer Module in pytorch Lightning
import torch
from fit.transformers_fit.PositionalEncoding2D import PositionalEncoding2D
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaBlock


class NewMamba(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conf = MambaConfig()
        conf.num_hidden_layers = 8
        conf.expand = 4
        conf.hidden_size = 256
        # conf.state_size = 32
        conf.intermediate_size = 512
        self.layers = torch.nn.ModuleList([MambaBlock(conf, layer_idx=x) for x in range(8)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
    
class SResTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords, flatten_order,
                 model_type='fast',
                 attention_type="full",
                 n_layers=8,
                 n_heads=8,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(SResTransformer, self).__init__()
        self.model_type = model_type
        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2) #shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords, #(r,phi)
            flatten_order=flatten_order,
            persistent=False
        ) 
        # Source: Mamba Repository
        self.encoder = NewMamba()
        self.encoder.to('cuda')
        
        self.predictor_amp = torch.nn.Linear(n_heads * d_query,1)
        self.predictor_phase = torch.nn.Linear(n_heads * d_query,1)

    def forward(self, x,causal = True):
        x = self.fourier_coefficient_embedding(x) #shape = 377,2 --> 377,128
        x = self.pos_embedding(x) #shape 377,128 --> 377,256
        
        if self.model_type == 'mamba':
            y_hat = self.encoder(x)
        
        # if self.model_type == 'torch':
        #     triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        #     mask = triangular_mask.additive_matrix_finite
        #     if not causal: mask = None
        #     y_hat = self.encoder(x, mask=mask)
        
        # if self.model_type == 'fast':
        #     triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        #     mask = triangular_mask
        #     if not causal: mask = None
        #     y_hat = self.encoder(x, attn_mask=mask)

        y_amp = self.predictor_amp(y_hat)
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1)
    
    def forward_inference(self, x,max_seq_length=378,causal = True): #(32,39,256)
        with torch.no_grad():
            x_hat = x.clone()
            for i in range(x.shape[1],max_seq_length):
                y_hat = self.forward(x_hat, causal = causal)
                x_hat = torch.cat([x_hat,y_hat[:,-1,:].unsqueeze(1)],dim = 1)
        print(x_hat[1].shape)
        assert x_hat.shape[1] == max_seq_length
        assert (x_hat[:,:x.shape[1]] == x).all()
        return x_hat
   