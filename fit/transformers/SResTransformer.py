# This file contains the implementation of the SResTransformer Module in pytorch Lightning
import torch
import torch.nn as nn
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
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
        self.model_type = model_type

        if self.model_type == 'torch':
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=n_heads * d_query, nhead=n_heads,batch_first=True), num_layers=n_layers)
        if self.model_type == 'fast' :
            self.encoder = TransformerEncoderBuilder.from_kwargs(
                attention_type=attention_type,
                n_layers=n_layers,
                n_heads=n_heads,
                feed_forward_dimensions=n_heads * d_query * 4,
                query_dimensions=d_query,
                value_dimensions=d_query,
                dropout=dropout,
                attention_dropout=attention_dropout
            ).get()       
        
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
    
    def forward_inference(self, x,input_seq_length, max_seq_length=377,causal = False):
        with torch.no_grad():
            x_hat = x
            for i in range(input_seq_length,378):
                y_hat = self.forward(x_hat, causal = causal)
                x_hat = torch.cat([x_hat,y_hat[:,-1,:].unsqueeze(1)],dim = 1)
        assert x_hat.shape[1] == 378
        # x_hat[:,:input_seq_length] = x
        return x_hat
   