import torch
from fit.transformers.masking import TriangularCausalMask
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


        self.encoder = torch.nn.Transformer(d_model = 256,nhead=8, num_encoder_layers=8,batch_first = True)


        
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
        x = torch.nn.Linear(256,377).to('cuda')(x)
        # triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        triangular_mask = torch.ones(377, 377,device = x.device).triu().T
        y_hat = self.encoder(x, mask=triangular_mask)
        y = torch.nn.Linear(377,256).to('cuda')(y_hat)
        y_amp = torch.tanh(self.predictor_amp(y))
        y_phase = torch.tanh(self.predictor_phase(y))
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
