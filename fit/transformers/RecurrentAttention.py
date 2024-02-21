import torch
import math
from torch.nn import Dropout, Linear, Module
from fit.transformers.events import EventDispatcher, QKVEvent


class RecurrentAttentionLayer(Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer.

    The only difference with the corresponding module is that this projects
    only one input and then calls the inner attention with the provided
    previous state.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, event_dispatcher=""):
        super(RecurrentAttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.inner_attention = attention
        self.query_projection = Linear(d_model, d_keys * n_heads)
        self.key_projection = Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = Linear(d_model_keys, d_values * n_heads)
        self.out_projection = Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None):
        """Apply attention to the passed in query/key/value after projecting
        them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            query: (N, D) The tensor containing the queries
            key: (N, D) The tensor containing the keys
            value: (N, D) The tensor containing the values
            state: The state varies depending on the inner attention implementation

        Returns
        -------
            The new value for each query as a tensor of shape (N, D).
        """

        # Project the queries/keys/values
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Reshape them into many heads and compute the attention
        N, D = query.shape
        H = self.n_heads
        new_value, state = self.inner_attention(
            query.view(N, H, -1),
            key.view(N, H, -1),
            value.view(N, H, -1),
            state
        )
        new_value = new_value.view(N, -1)

        # Project the output and return
        return self.out_projection(new_value), state


class RecurrentFullAttention(Module):
    """Implement the full softmax attention as a recurrent module.

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
    def __init__(self, softmax_temp=None, attention_dropout=0.1):
        super(RecurrentFullAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)

    def forward(self, query, key, value, state=None):

        # Extract some shapes and compute the temperature
        N, H, E = query.shape
        _, _, D = value.shape
        softmax_temp = self.softmax_temp or 1./math.sqrt(E)

        # Aggregate the list of keys and values
        if state is not None:
            keys, values = state
            keys = torch.cat([keys, key[:, :, None]], dim=2)
            values = torch.cat([values, value[:, :, None]], dim=2)
        else:
            keys = key[:, :, None]
            values = value[:, :, None]

        # Compute the unnormalized attention
        QK = torch.einsum("nhe,nhse->nhs", query, keys)

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhs,nhsd->nhd", A, values).contiguous()

        # Let the world know of the attention matrix

        # Make sure that what we return is contiguous
        return V, [keys, values]
 