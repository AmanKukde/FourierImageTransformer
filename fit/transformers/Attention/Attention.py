# This file is a copy of the original file from the fast-transformers library
"""Implement the full attention similar to the one implemented by PyTorch's
MultiHeadAttention module. Note that this module is to be used in conjuction
with the `fast_transformers.attention.attention_layer.AttentionLayer` in order
to work."""

from math import sqrt

import torch
from torch.nn import Dropout, Linear, Module

class AttentionLayer(Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads) #32
        d_values = d_values or (d_model//n_heads) #32
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads) #256 --> 256
        self.key_projection = Linear(d_model_keys, d_keys * n_heads) #256 --> 256
        self.value_projection = Linear(d_model_keys, d_values * n_heads) #256 --> 256
        self.out_projection = Linear(d_values * n_heads, d_model) #256 --> 256
        self.n_heads = n_heads #8

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)
   
class FullAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(FullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        ## Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        # softmax_temp = self.softmax_temp or 1./sqrt(E)

        ##Scale the queries instead of applying the softmax temperature to the
        ##dot products

        # queries = queries * softmax_temp
        
        ## Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)


        # Make sure that what we return is contiguous
        return V.contiguous()