from json import decoder
import torch
from fit.transformers.masking import TriangularCausalMask,FullMask
from fit.transformers.EncoderBlock import EncoderBlock
from fit.transformers.RecurrentEncoderBlock import RecurrentEncoderBlock
from fit.transformers.PositionalEncoding2D import PositionalEncoding2D
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.transformer import _generate_square_subsequent_mask


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

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model // 2)#shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords,#(r,phi)
            flatten_order=flatten_order,
            persistent=False
        ) 

        # self.encoder = EncoderBlock(d_model=d_model,d_query = d_query, n_layers=n_layers, n_heads=n_heads, dropout = dropout,attention_dropout=attention_dropout)
        encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=8)
        decoder_layer = TransformerDecoderLayer(d_model=256, nhead=8,batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=8)


        self.predictor_amp = torch.nn.Linear(
            n_heads * d_query,
            1
        )
        self.predictor_phase = torch.nn.Linear(
            n_heads * d_query,
            1
        )

    def forward(self, x,y):
        x = torch.tanh(self.fourier_coefficient_embedding(x))  #shape = 377,2 --> 377,128
        x = self.pos_embedding(x) #shape 377,128 --> 377,256
        y = torch.tanh(self.fourier_coefficient_embedding(y))  #shape = 377,2 --> 377,128
        y = self.pos_embedding(y) #shape 377,128 --> 377,256
        triangular_mask =  _generate_square_subsequent_mask(x.shape[1], device=x.device)
        y_hat = self.encoder(x,triangular_mask, is_causal = True)
        y_hat = self.decoder(tgt = y, memory = y_hat,tgt_mask = triangular_mask)
        y_amp = torch.tanh(self.predictor_amp(y_hat))
        y_phase = torch.tanh(self.predictor_phase(y_hat))
        return torch.cat([y_amp, y_phase], dim=-1)
    
    def forward_i(self, x,input_seq_length=100):
        self.encoder.eval()
        self.decoder.eval()
        self.pos_embedding.eval()
        self.fourier_coefficient_embedding.eval()
        self.predictor_amp.eval()
        self.predictor_phase.eval()
        padded_input = torch.zeros(x.shape[0],377,256).to(x.device)
        with torch.no_grad():
            x_hat = torch.tanh(self.fourier_coefficient_embedding(x))
            x_hat = self.pos_embedding(x_hat)
            padded_input[:,:input_seq_length] = x_hat
            memory = self.encoder(padded_input)
            for i in range(input_seq_length,377):
                y_hat = self.decoder(padded_input,memory)
                padded_input[:,i,:] =  y_hat[:,i-1,:]  #97th element but 96th index
            
            output = padded_input
            y_amp = torch.tanh(self.predictor_amp(output))
            y_phase = torch.tanh(self.predictor_phase(output))
            y_amp[:,:input_seq_length] = x[:,:input_seq_length,0].unsqueeze(-1)
            y_phase[:,:input_seq_length] = x[:,:input_seq_length,1].unsqueeze(-1)
        return torch.cat([y_amp, y_phase], dim=-1)
    
    def forward_ii(self, x,input_seq_length=100):
        self.encoder.eval()
        self.pos_embedding.eval()
        self.fourier_coefficient_embedding.eval()
        self.predictor_amp.eval()
        self.predictor_phase.eval()
        with torch.no_grad():
            x_hat = torch.tanh(self.fourier_coefficient_embedding(x))
            x_hat = self.pos_embedding(x_hat)

            
            print(x_hat.shape)
            for i in range(input_seq_length,377):
                y_hat = self.encoder(x_hat,)
                x_hat = torch.cat([x_hat,y_hat[:,-1,:].unsqueeze(1)],dim = 1) #97th element but 96th index
                # print(x_hat.shape)
            output = x_hat
            y_amp = torch.tanh(self.predictor_amp(output))
            y_phase = torch.tanh(self.predictor_phase(output))
            y_amp[:,:input_seq_length] = x[:,:input_seq_length,0].unsqueeze(-1)
            y_phase[:,:input_seq_length] = x[:,:input_seq_length,1].unsqueeze(-1)
        return torch.cat([y_amp, y_phase], dim=-1)
    


