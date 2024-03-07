import torch
import sys
sys.path.append('/home/aman.kukde/Projects/FourierImageTransformer/fast_transformers/')
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D


class SResTransformerTrain(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords, flatten_order,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(SResTransformerTrain, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding = PositionalEncoding2D(
            d_model // 2,
            coords=coords,
            flatten_order=flatten_order,
            persistent=False
        )

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
        
        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

        self.encoder.require_grad= True
        self.predictor_amp.require_grad= False
        self.predictor_phase.require_grad= False
        self.fourier_coefficient_embedding.require_grad= False
        self.pos_embedding.require_grad= False

    def forward(self, x):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding(x)
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.encoder(x, attn_mask=triangular_mask)
        y_amp = self.predictor_amp(y_hat)
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1)
    
    # def forward_i(self, x):
    #     x = self.fourier_coefficient_embedding(x)
    #     x = self.pos_embedding(x)
    #     full_sequence = torch.zeros(x.shape[0],377,x.shape[-1])
    #     full_sequ:,:self.input_seq_length,:] = x
    #     mask_scaffold  = torch.full((377,377),-float('Inf')).to(x.device)
    #     mask_scaffold[:self.input_seq_length,:self.input_seq_length] = 0
       
    #     return torch.cat([y_amp, y_phase], dim=-1)
    
    def forward_i(self, x,input_seq_length=100):
        self.encoder.eval()
        self.pos_embedding.eval()
        self.fourier_coefficient_embedding.eval()
        padded_input = torch.zeros(x.shape[0],377,256).to(x.device)
        mask = torch.full((377,377),bool(False)).to(x.device)
        with torch.no_grad():
            x_hat = self.fourier_coefficient_embedding(x)
            x_hat = self.pos_embedding(x_hat)
            padded_input[:,:input_seq_length] = x_hat
            for i in range(input_seq_length,377-1):
                mask[:i,:i] = True
                fast_mask = FullMask(mask = mask, device=x.device)
                y_hat = self.encoder(padded_input,attn_mask = fast_mask)
                padded_input[:,i,:] =  y_hat[:,i,:]  #97th element but 96th index
            output = padded_input
            y_amp = self.predictor_amp(output)
            y_phase = torch.tanh(self.predictor_phase(output))
            # y_amp[:,:input_seq_length] = x[:,:input_seq_length,0].unsqueeze(-1)
            # y_phase[:,:input_seq_length] = x[:,:input_seq_length,1].unsqueeze(-1)
            return torch.cat([y_amp, y_phase], dim=-1)

