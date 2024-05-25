# This file contains the implementation of the SResTransformer Module in pytorch Lightning
from os import preadv
from importlib_metadata import Prepared
from sympy import preorder_traversal
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
                 shells = 6,
                 dft_shape = None,
                 fc_per_ring=None,
                 no_of_sectors = 20,
                 semi_circle_only_flag = False,
                 attention_dropout=0.1):
        super(SResTransformer, self).__init__()
        self.model_type = model_type
        self.fc_per_ring  = fc_per_ring
        self.shells = shells
        if self.shells == None:
            self.shells = len(self.fc_per_ring.keys())
        self.dft_shape = dft_shape
        self.interpolation_size = interpolation_size
        self.dst_flatten_order = flatten_order
        self.d_model = d_model
        self.no_of_sectors = no_of_sectors
        self.fourier_coefficient_embedding = torch.nn.Linear(2*self.interpolation_size//self.no_of_sectors, d_model // 2) #shape = (2,N/2)
        self.semicircle_only_flag = semi_circle_only_flag
        self.embed_amps = torch.nn.Linear(self.no_of_sectors, d_model // 4) #shape = (5 --> 64)
        self.embed_phi = torch.nn.Linear(self.no_of_sectors, d_model // 4) #shape = (5 --> 64)
        
        self.pos_embedding = PositionalEncoding2D(
            d_model // 2, #F/2
            coords=coords, #(r,phi)
            flatten_order=flatten_order,
            persistent=False
        )
        self.model_type = model_type

        if self.model_type == 'mamba':
            conf = MambaConfig()
            conf.num_hidden_layers = n_layers
            conf.expand = 4
            conf.hidden_size =d_model 
            conf.intermediate_size =d_model * 2
            self.encoder = MambaModel(conf)

        if self.model_type == 'torch':
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,batch_first=True), num_layers=n_layers)
        if self.model_type == 'fast' :
            self.encoder = TransformerEncoderBuilder.from_kwargs(
                attention_type=attention_type,
                n_layers=n_layers,
                n_heads=n_heads,
                feed_forward_dimensions=d_model * 4,
                query_dimensions=d_query,
                value_dimensions=d_query,
                dropout=dropout,
                attention_dropout=attention_dropout
            ).get() 


        self.encoder.to('cuda')
        
        self.predictor_amp = torch.nn.Linear(d_model//2,self.no_of_sectors)
        self.predictor_phase = torch.nn.Linear(d_model//2,self.no_of_sectors)
        
    
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
            permuted_fc = fc.clone().permute(0,2,1)#batch,2,tokens #interpolation size should be multiple of self.no_of_sectors for easy calculation #14*3.14 = 44-->50
    
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
        largest_semicircle = interpolated_fc[...,:self.dft_shape[1],:] #32,2,19,50 --> 32,2,14,50
        return largest_semicircle
    
    def get_amp_phase(self,input_tensor):
        amps = input_tensor[...,0].reshape(*input_tensor.shape[:1],-1,self.no_of_sectors) #32,19,50,2 --> 32,19,5,10,1
        phases = input_tensor[...,1].reshape(*input_tensor.shape[:1],-1,self.no_of_sectors) #32,19,50,2 --> 32,19,5,10,1
        return amps,phases

    def get_lowres_input(self, highres_input):
        s = (self.shells * 2 * self.interpolation_size //100)
        lowres_input = highres_input[:,:,:s,:]
        lowres_input = lowres_input.permute(0,2,1,-1) 
        lowres_input = lowres_input.reshape(highres_input.shape[0],-1,2*(self.interpolation_size//self.no_of_sectors))
        return lowres_input
    
    def embed_tensor(self, input_tensor):
        amps, phases = self.get_amp_phase(input_tensor) #32,19,50,2 --> (32,19,5,10,1), (32,19,5,10,1)
        tokenised_amps = self.embed_amps(amps) #32,19,5,10,1 --> 32,19,5,64
        tokenised_phases = self.embed_phi(phases) #32,19,5,10,1 --> 32,19,5,64
        tokenised_tensor =  torch.cat([tokenised_amps,tokenised_phases],dim = -1) #32,19,5,64 + 32,19,5,64 --> 32,19,5,128
        tokenised_tensor = tokenised_tensor.reshape(tokenised_tensor.shape[0],-1,self.d_model//2) #32,19,5,128 --> 32,95,128

        return tokenised_tensor
        
        

    def forward(self, fc):

        interpolated_fc = self.get_interpolated_rings(fc) # 32,378,2 --> 32,2,19,50
        
        # if self.semicircle_only_flag:
        #     input_ = self.get_largest_semicircle(interpolated_fc).permute(0,2,-1,1)  #32,19,50,2 --> 32,14,50,2
        
        # else:
        input_ = interpolated_fc.clone().permute(0,2,-1,1)  #32,19,50,2 
        
        embedded_tensor = self.embed_tensor(input_) #32,19,50,2 --> 32,95,128

        embedded_tensor_positional = self.pos_embedding(embedded_tensor) #32,95,128 --> 32,95,256 
        
        enc_output = self.encoder_forward(embedded_tensor_positional) #32,95,256 --> 32,95,256
        
        y_amp = torch.tanh(self.predictor_amp(enc_output[...,:enc_output.shape[-1]//2])) #32,95,128 --> 32,95,10,1
        y_phase = torch.tanh(self.predictor_phase(enc_output[...,enc_output.shape[-1]//2:])) #32,95,128 --> 32,95,10,1
        
        enc_output_amp_phi =  torch.stack([y_amp, y_phase], dim=-1) #32,95,10,1 + 32,95,10,1 --> 32,95,10,2
        
        #since we gave model the input of 95 tokens, we predict 95 tokens but these are from 2nd to last+1 
        # so we will concat the first token of input with the second token to second last of the output and discard the last token thus predicted.
        ## Need to reshape input_ so that the shapes match each other to concat. 
        pred = torch.cat([input_.reshape(*input_.shape[:1],-1,self.no_of_sectors,2)[:,:1,:,:],enc_output_amp_phi[:,:-1,:,:]], dim=1) #32,1,10,2 + 32,94,10,2 --> 32,95,10,2
        
        # if self.semicircle_only_flag:
        #     pred = self.get_full_plane_from_sectors(pred,interpolated_fc)
        # else:
        pred = pred.reshape(pred.shape[0],-1,self.interpolation_size,2).permute(0,3,1,2) #32,95,10,2 --> 32,19,50,2 --> 32,2,19,50
        
        model_output = self.get_de_interpolated_rings(pred) #32,2,19,50 --> 32,378,2

        return model_output #32,378,2
    

    def get_full_plane_from_sectors(self,output,interpolated_fc):
        output = output.reshape(interpolated_fc.shape[0],-1,2,self.interpolation_size) #batch,2, ring,self.interpolation_size
        output = output.permute(0,2,1,3)
        if self.semicircle_only_flag:
            output = torch.cat([output, interpolated_fc[...,self.dft_shape[1]:,:]],dim = -2) #batch,2,interpolation_size,ring
        return output
    
    def get_de_interpolated_rings(self,output):
        final_output = []
        for r in self.fc_per_ring.keys():
            final_output.append(interpolate(output[:,:,r,:],self.fc_per_ring[r], mode ="nearest-exact"))
        final_output = torch.cat(final_output,dim = -1).permute(0,2,1) #batch,378,2  
        return final_output
    
    def forward_i(self, fc, max_seq_length = None):
        
        interpolated_fc = self.get_interpolated_rings(fc) # 32,378,2 --> 32,2,19,50

        if max_seq_length == None:
            max_seq_length = len(list(self.fc_per_ring.keys()))*self.interpolation_size//self.no_of_sectors #95

        lowres_input = self.get_lowres_input(interpolated_fc) #32,19,50,2 --> 32,95,10,2

        embedded_tensor = self.embed_tensor(lowres_input) #32,95,10,2 --> 32,95,256

        embedded_tensor_positional = self.pos_embedding(embedded_tensor) #32,95,256
        
        encoder_output = self.encoder_i(embedded_tensor_positional) #32,10,256--> 32,95,256

        y_amp = torch.tanh(self.predictor_amp(encoder_output[...,:encoder_output.shape[-1]//2])) #32,95,128 --> 32,95,10,1
        y_phase = torch.tanh(self.predictor_phase(encoder_output[...,encoder_output.shape[-1]//2:])) #32,95,128 --> 32,95,10,1
  
        pred =  torch.stack([y_amp, y_phase], dim=-1) #32,95,10,1 + 32,95,10,1 --> 32,95,10,2
        
        # if self.semicircle_only_flag:
        #     pred = self.get_full_plane_from_sectors(pred,interpolated_fc)
        # else:
        pred = pred.reshape(pred.shape[0],-1,self.interpolation_size,2).permute(0,3,1,2) #32,95,10,2 --> 32,19,50,2 --> 32,2,19,50
        
        model_output = self.get_de_interpolated_rings(pred) #32,2,19,50 --> 32,378,2

        return model_output #32,378,2
    
    def encoder_i(self, x,max_seq_length = 95): #(32,39,256)
        with torch.no_grad():
            x_hat = x.clone()
            for i in range(x.shape[1],max_seq_length):
                y_hat = self.encoder_forward(x_hat) #32,39,256
                x_hat = torch.cat([x_hat,y_hat[:,-1:,:]],dim = 1) #32,40,256
                
        print(x_hat.shape) #32,95,256
        assert x_hat.shape[1] == max_seq_length #95
        return x_hat #32,95,256
    
    def encoder_forward(self, x):
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

        return y_hat
        
    


    
    