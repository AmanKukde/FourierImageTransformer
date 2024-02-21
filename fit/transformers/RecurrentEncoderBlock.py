import warnings
import torch
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
from fit.transformers.events import EventDispatcher, IntermediateOutput
from fit.transformers.RecurrentAttention import RecurrentFullAttention,RecurrentAttentionLayer
# from utils import check_state


class RecurrentTransformerEncoderLayer(Module):
    """Attention to the previous inputs and feed forward with skip connections.

    This transformer encoder layer is the recurrent dual of
    fast_transformers.transformers.TransformerEncoderLayer . The results should
    be identical given the same inputs and a lower triangular mask.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(RecurrentTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None):
        """Apply the transformer encoder to the input x using the provided
        state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor
            state: The state can vary depending on the attention implementation
        """

        # Run the self attention and add it to the input
        # if state is not None:
        #     k,v = state
        # else:
        #     k = x; v = x
        x2, state = self.attention(x,x,x,state) #TODO: check if this is the correct way to use the attention
        x = x + self.dropout(x2)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y), state

class RecurrentTransformerEncoder(Module):
    """RecurrentTransformerEncoder is a sequence of
    RecurrentTransformerEncoderLayer instances.

    RecurrentTransformerEncoder keeps a separate state per
    RecurrentTransformerEncoderLayer.

    Arguments
    ---------
        layers: list, RecurrentTransformerEncoderLayer instances or instances
                that implement the same interface
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(RecurrentTransformerEncoder, self).__init__()
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, state=None):
        """Apply all recurrent transformer layers to the input x using the
        provided state.

        Arguments
        ---------
            x: The input features of shape (N, E) where N is the batch size and
               E is d_model passed in the constructor of each recurrent
               transformer encoder layer
            state: A list of objects to be passed to each recurrent
                   transformer encoder layer
        """
        if state is None:
            state = [None]*len(self.layers)

        # Apply all the transformers
        for i, layer in enumerate(self.layers):
            x, s = layer(x, state[i])
            state[i] = s
            self.event_dispatcher.dispatch(IntermediateOutput(self, x))

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, state
    
class RecurrentEncoderBlock(Module):
    def __init__(self,d_model, n_layers, n_heads, dropout,attention_dropout,*args, **kwargs) -> None:
        super().__init__()
        
        self.n_layers = n_layers
        self.attention = RecurrentAttentionLayer(RecurrentFullAttention(), d_model, n_heads)
        self.encoder_layer = RecurrentTransformerEncoderLayer(attention = self.attention, d_model = d_model, d_ff=None, dropout=0.1, activation="relu")
        self.layers = ModuleList([self.encoder_layer for _ in range(n_layers)])
    
    def forward(self, x, state = None, mask=None):
        y,state = self.layers[0].forward(x,state)
        for i in range(1,self.n_layers):
            y,state = self.layers[i].forward(y,state)
        return y,state