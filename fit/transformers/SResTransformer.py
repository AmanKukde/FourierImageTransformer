import torch
from mamba_ssm import Mamba
from fit.transformers.masking import TriangularCausalMask,FullMask
from fit.transformers.EncoderBlock import EncoderBlock
from fit.transformers.RecurrentEncoderBlock import RecurrentEncoderBlock
from fit.transformers.PositionalEncoding2D import PositionalEncoding2D


class SResTransformerTrain(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords, flatten_order,
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(SResTransformerTrain, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2) #shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords,#(r,phi)
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
        
        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

    def forward(self, x):
        x = self.fourier_coefficient_embedding(x)  #shape = 377,2 --> 377/2,2 = 189,2
        x = self.pos_embedding(x) #shape = 189,2 --> 377,2
        
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.encoder(x, mask=triangular_mask)
        y_amp = torch.tanh(self.predictor_amp(y_hat))
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1)
    
    def forward_i(self, x,input_seq_length=100):
        self.encoder.eval()
        self.pos_embedding.eval()
        self.fourier_coefficient_embedding.eval()
        padded_input = torch.zeros(x.shape[0],377,256).to(x.device)
        with torch.no_grad():
            x_hat = self.fourier_coefficient_embedding(x)
            x_hat = self.pos_embedding(x_hat)
            padded_input[:,:input_seq_length] = x_hat
            for i in range(input_seq_length,377-1):
                fast_mask = TriangularCausalMask(padded_input.shape[1], device=x.device)
                y_hat = self.encoder(padded_input, mask=fast_mask)
                padded_input[:,i,:] =  y_hat[:,i-1,:]  #97th element but 96th index
            output = padded_input
            y_amp = self.predictor_amp(output)
            y_phase = torch.tanh(self.predictor_phase(output))
            y_amp[:,:input_seq_length] = x[:,:input_seq_length,0].unsqueeze(-1)
            y_phase[:,:input_seq_length] = x[:,:input_seq_length,1].unsqueeze(-1)
            return torch.cat([y_amp, y_phase], dim=-1)
    


class SResTransformerPredict(torch.nn.Module):

    def __init__(self, d_model, coords, flatten_order, 
                 n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1,
                 attention_dropout=0.1):
        super(SResTransformerPredict, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)

        self.pos_embedding = PositionalEncoding2D(
            d_model // 2,
            coords=coords,
            flatten_order=flatten_order,
            persistent=False
        )

        self.encoder =  RecurrentEncoderBlock(d_model, n_layers, n_heads, dropout, attention_dropout)

        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

    def forward(self, x, i=0, state=None):
        x = x.view(x.shape[0], -1)
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding.forward_i(x, i)
        y_hat, state = self.encoder.forward(x, state)
        y_amp = torch.tanh(self.predictor_amp(y_hat))
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1), state

