# This file contains the implementation of the SResTransformer Module in pytorch Lightning
import torch
from fit.transformers_fit.PositionalEncoding2D import PositionalEncoding2D
from fast_transformers.masking import TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder
from transformers import MambaConfig,MambaModel
import numpy as np
from torch.nn.functional import avg_pool3d, interpolate
import math

   
class SResTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords, flatten_order,
                 model_type='fast',
                 attention_type="full",
                 interpolation_size=50,
                 n_layers=8,
                 n_heads=8,
                 d_query=32,
                 dropout=0.1,
                 dft_shape = None,
                 fc_per_ring=None,
                 attention_dropout=0.1):
        super(SResTransformer, self).__init__()
        self.model_type = model_type
        self.fc_per_ring  = fc_per_ring
        self.dft_shape = dft_shape
        self.interpolation_size = interpolation_size
        self.dst_flatten_order = flatten_order
        self.fourier_coefficient_embedding = torch.nn.Linear(10, d_model // 2) #shape = (2,N/2)
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords, #(r,phi)
            flatten_order=flatten_order,
            persistent=False
        ) 
        self.model_type = model_type

        # if self.model_type == 'mamba':
        #     conf = MambaConfig()
        #     conf.num_hidden_layers = 24
        #     conf.expand = 4
        #     conf.hidden_size = 2
        #     conf.intermediate_size =128
        #     self.encoder = MambaModel(conf)

        if self.model_type == 'mamba':
            conf = MambaConfig()
            conf.num_hidden_layers = n_layers
            conf.expand = 4
            conf.hidden_size = d_query*n_heads
            conf.intermediate_size = d_query*n_heads*2
            self.encoder = MambaModel(conf)

        if self.model_type == 'torch':
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=n_heads * d_query, nhead=n_heads,batch_first=True), num_layers=n_layers)
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

        # self.FullyConnectedNN = torch.nn.Sequential(
        #     torch.nn.Linear(2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, 10),
        # ).to('cuda')

        self.encoder.to('cuda')
        
        self.predictor_amp = torch.nn.Linear(n_heads * d_query,interpolation_size//10)
        # self.predictor_amp = torch.nn.Linear(n_heads * d_query,1)
    
        self.predictor_phase = torch.nn.Linear(n_heads * d_query,interpolation_size//10)
        # self.predictor_phase = torch.nn.Linear(n_heads * d_query,1)
    
    def get_interpolated_rings(self, fc):
            """
            Embeds the sectors of the largest semicircle of the input fourier coefficients tensor.
            Returned x is from 0 to -2 token of the input tensor. [:,:-1]

            Args:
                fc (torch.Tensor): Input feature map tensor of shape (batch_size, num_channels, height, width).

            Returns:
                torch.Tensor: Embedded sectors tensor of shape (batch_size, num_sectors, embedding_size).
                torch.Tensor: Interpolated input tensor of shape (batch_size, num_channels, interpolation_size, num_rings).
            """
            permuted_fc = fc.clone().permute(0,2,1)#batch,2,tokens
                        #interpolation size should be multiple of 10 for easy calculation #14*3.14 = 44-->50
                        
            # interpolated_fc = torch.zeros(size=(fc.shape[0],2,self.interpolation_size,len(model.sres.fc_per_ring.keys()))).to('cuda')
            r = 0
            selected_ring = permuted_fc[:,:,:self.fc_per_ring[r]]
            permuted_fc = permuted_fc[:,:,self.fc_per_ring[r]:]  
            interpolated_fc = interpolate(selected_ring,self.interpolation_size, mode = 'nearest-exact').unsqueeze(2)

            for r in list(self.fc_per_ring.keys())[1:]:
                selected_ring = permuted_fc[:,:,:self.fc_per_ring[r]]
                permuted_fc = permuted_fc[:,:,self.fc_per_ring[r]:]  
                interpolated_fc = torch.cat([interpolated_fc,interpolate(selected_ring,self.interpolation_size, mode = 'nearest-exact').unsqueeze(2)], axis = 2)

            return interpolated_fc

    def get_largest_semicircle(self,interpolated_fc):
        largest_semicircle = interpolated_fc[...,:self.dft_shape[1],:]
        largest_semicircle = largest_semicircle.permute(0,2,1,-1)  #batch,ring,2,interpolation_size
        largest_semicircle = largest_semicircle.reshape(interpolated_fc.shape[0],-1,2*(self.interpolation_size//10))
        return largest_semicircle
        


    def forward(self, fc):

        interpolated_fc = self.get_interpolated_rings(fc)

        largest_semicircle = self.get_largest_semicircle(interpolated_fc)

        pred = self.model_forward(largest_semicircle[:,:-1,:]) #batch,tokens,2
        
        output = torch.cat([largest_semicircle[:,:1,:],pred],dim = 1)
        
        output = self.get_full_plane_from_sectors(output,interpolated_fc)
        
        final_output = self.get_de_interpolated_rings(output)


        return final_output
    

    def get_full_plane_from_sectors(self,output,interpolated_fc):
         #batch,ring,2,interpolation_size
        output = output.reshape(interpolated_fc.shape[0],-1,2,self.interpolation_size) #batch,2, ring,self.interpolation_size
        output = output.permute(0,2,1,3)
        output = torch.cat([output, interpolated_fc[...,self.dft_shape[1]:,:]],dim = -2) #batch,2,interpolation_size,ring
        return output
    
    def get_de_interpolated_rings(self,output):
        final_output = []
        for r in self.fc_per_ring.keys():
            final_output.append(interpolate(output[:,:,r,:],self.fc_per_ring[r], mode ="nearest-exact"))
        final_output = torch.cat(final_output,dim = -1).permute(0,2,1) #batch,378,2  
        return final_output
    
    def model_forward_inference(self, x,max_seq_length=378): #(32,39,256)
        with torch.no_grad():
            x_hat = x.clone()
            for i in range(x.shape[1],max_seq_length):
                y_hat = self.forward(x_hat)
                x_hat = torch.cat([x_hat,y_hat[:,-1,:].unsqueeze(1)],dim = 1)
        print(x_hat[1].shape)
        assert x_hat.shape[1] == max_seq_length
        assert (x_hat[:,:x.shape[1]] == x).all()
        return x_hat
    
    def model_forward(self, x):

        x = self.fourier_coefficient_embedding(x) #shape = 377,2 --> 377,128
        x = self.pos_embedding(x) #shape 377,128 --> 377,256
        
        if self.model_type == 'mamba':
            y_hat = self.encoder(inputs_embeds = x)
            y_hat = y_hat.last_hidden_state
        if self.model_type == 'torch':
            triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
            mask = triangular_mask.additive_matrix_finite
            y_hat = self.encoder(x, mask=mask)
        
        if self.model_type == 'fast':
            triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
            mask = triangular_mask
            y_hat = self.encoder(x, attn_mask=mask)

        y_amp = self.predictor_amp(y_hat)
        y_phase = torch.tanh(self.predictor_phase(y_hat))

        return torch.stack([y_amp, y_phase], dim=-1).reshape(x.shape[0], x.shape[1], -1) #shape 377,5,2 --> 377,5,2 --> 377,10
    
    def get_rings_using_RNN(self,output):
        RNN = torch.nn.RNN(input_size = 2,hidden_size = 2,batch_first = True)
        final_output = []
        for r in self.fc_per_ring.keys():
            tokens, hidden = RNN(output[:,:,r,:])
            for i in range(1,self.fc_per_ring[r]):
                tokens.append(RNN(tokens))

            final_output.append(interpolate(output[:,:,r,:],self.fc_per_ring[r]))
        final_output = torch.cat(final_output,dim = -1).permute(0,2,1) #batch,378,2    
        return final_output