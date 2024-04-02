# This file contains the implementation of the SResTransformer Module in pytorch Lightning
import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from mamba_ssm import Mamba
from fit.transformers.masking import TriangularCausalMask,FullMask
from fit.transformers.EncoderBlock import EncoderBlock
from fit.transformers.RecurrentEncoderBlock import RecurrentEncoderBlock
from fit.transformers.PositionalEncoding2D import PositionalEncoding2D

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

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2) #shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords, #(r,phi)
            flatten_order=flatten_order,
            persistent=False
        ) 
        # Source: Mamba Repository


        batch, length, dim = 2, 64, 16
        x = torch.randn(batch, length, dim).to("cuda")
        model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to("cuda")
        
        self.encoder = EncoderBlock(d_model=d_model,d_query = d_query, n_layers=n_layers, n_heads=n_heads, dropout = dropout,attention_dropout=attention_dropout)
        
        self.predictor_amp = torch.nn.Linear(n_heads * d_query,1)
        self.predictor_phase = torch.nn.Linear(n_heads * d_query,1)

    def forward(self, x,causal = True):
        x = self.fourier_coefficient_embedding(x) #shape = 377,2 --> 377,128
        x = self.pos_embedding(x) #shape 377,128 --> 377,256
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        
        if self.model_type == 'torch':
            mask = triangular_mask.additive_matrix_finite
            if not causal: mask = None
            y_hat = self.encoder(x, mask=mask)
        
        if self.model_type == 'fast':
            mask = triangular_mask
            if not causal: mask = None
            y_hat = self.encoder(x, attn_mask=mask)

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
   