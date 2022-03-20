import torch
from fast_transformers.builders import TransformerDecoderBuilder, TransformerEncoderBuilder

from fit.transformers.PositionalEncoding2D import PositionalEncoding2D


class TRecTransformer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 coords_sinogram, flatten_order_sinogram,
                 coords_target, flatten_order_target,
                 attention_type="linear",
                 n_layers=4,
                 n_heads=4,
                 d_query=32,
                 dropout=0.1,
                 attention_dropout=0.1):
        super(TRecTransformer, self).__init__()

        self.fourier_coefficient_embedding = torch.nn.Linear(2, d_model)

        self.pos_embedding_input_projections = PositionalEncoding2D(
            d_model,
            coords=coords_sinogram,
            flatten_order=flatten_order_sinogram,
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

        self.pos_embedding_target = PositionalEncoding2D(d_model, coords=coords_target,
                                                         flatten_order=flatten_order_target)

        self.decoder = TransformerDecoderBuilder.from_kwargs(
            self_attention_type=attention_type,
            cross_attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads * d_query * 4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            attention_dropout=attention_dropout
        ).get()

        self.predictor = torch.nn.Linear(
            n_heads * d_query,
            2
        )
        # self.predictor_phase = torch.nn.Linear(
        #     n_heads * d_query,
        #     1
        # )

    def forward(self, x, target_fc):
        x = self.fourier_coefficient_embedding(x)
        x = self.pos_embedding_input_projections(x)
        z = self.encoder(x, attn_mask=None)

        x_ = self.fourier_coefficient_embedding(target_fc)
        x_ = self.pos_embedding_target(x_)
        y_hat = self.decoder(x_, z)
        y_hat = self.predictor(y_hat)
        y_hat = torch.cat([y_hat[...,0], torch.tanh(y_hat[...,1])], dim=-1)

        # y_amp = self.predictor_amp(y_hat)
        # y_phase = torch.tanh(self.predictor_phase(y_hat))
        # y_hat = torch.cat([y_amp, y_phase], dim=-1)

        return y_hat
