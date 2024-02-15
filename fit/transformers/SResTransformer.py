import torch
from fit.transformers.masking import TriangularCausalMask
from fit.transformers.my_encoder import MyEncoder
from fit.transformers.my_recurrent_encoder import MyRecEncoder
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

        self.encoder = MyEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout = dropout,attention_dropout=attention_dropout)
        
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
        
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device).float_matrix
        y_hat = self.encoder(x, mask=triangular_mask)
        y_amp = self.predictor_amp(y_hat)
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1)


class SResTransformerPredict(torch.nn.Module):

    def __init__(self, d_model, coords, flatten_order,
                 attention_type="full", n_layers=4, n_heads=4,
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

        self.encoder =  MyRecEncoder(d_model, n_layers, n_heads, dropout)

        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

    def forward(self, x, i=0, memory=None):
        x = x.view(x.shape[0], -1)
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding.forward_i(x, i)
        y_hat, memory = self.encoder(x, memory)
        y_amp = self.predictor_amp(y_hat)
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1), memory
