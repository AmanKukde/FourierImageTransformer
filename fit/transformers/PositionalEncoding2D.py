import math
import torch


class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, d_model, y_coords, x_coords, flatten_order, dropout=0.0, persistent=False):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = self.positional_encoding_2D(self.d_model, y_coords, x_coords)
        pe = torch.movedim(pe, 0, -1).unsqueeze(0)
        pe = pe[:, flatten_order]

        self.register_buffer('pe', pe, persistent=persistent)

    def positional_encoding_2D(self, d_model, y_coords, x_coords):
        pe = torch.zeros(d_model, y_coords.shape[0])

        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[0:d_model:2, :] = torch.sin(y_coords.unsqueeze(-1) * div_term).permute(-1, 0)
        pe[1:d_model:2, :] = torch.cos(y_coords.unsqueeze(-1) * div_term).permute(-1, 0)
        pe[d_model::2, :] = torch.sin(x_coords.unsqueeze(-1) * div_term).permute(-1, 0)
        pe[d_model + 1::2, :] = torch.cos(x_coords.unsqueeze(-1) * div_term).permute(-1, 0)

        return pe

    def forward(self, x):
        pos_embedding = self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

    def forward_i(self, x, i):
        pos_embedding = self.pe[0, i:i + 1]
        x = torch.cat([x, pos_embedding.expand_as(x)], dim=1)
        return self.dropout(x)
