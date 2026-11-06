import torch
import torch.nn as nn

from mamba_ssm import Mamba2

import math

def build_encoder(
    name,
    input_dim,
    sequence_length,
    d_model,
    **kwargs,
):
    if name == "transformer":
        return TimeSeriesTransformerEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length,
            d_model=d_model,
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            use_cls=kwargs.get("use_cls", False),
        )

    elif name == "mamba":
        return MambaEncoder(
            input_dim=input_dim,
            sequence_length=sequence_length,
            d_model=d_model,
            num_layers=kwargs["num_layers"],
            d_state=kwargs.get("mamba_d_state", 64),
            d_conv=kwargs.get("mamba_d_conv", 4),
            expand=kwargs.get("mamba_expand", 2),
            headdim=kwargs.get("mamba_headdim", 64),
        )
    # elif name == "dlinear":
    #     return DLinearEncoder(
    #         input_dim=input_dim,
    #         sequence_length=sequence_length,
    #         d_model=d_model,
    #         kernel_size=kwargs.get("kernel_size", 25),
    #     )

    # elif name == "patchtst":
    #     return PatchTSTEncoder(...)

    # elif name == "timemixer":
    #     return TimeMixerEncoder(...)

    # elif name == "micn":
    #     return MICNEncoder(...)

    # elif name == "mamba":
    #     return MambaEncoder(...)

    else:
        raise ValueError(f"Unknown encoder: {name}")



class PositionalEncoding(nn.Module):
    """Classical Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


    
class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model,
        num_heads,
        num_layers,
        dropout,
        use_cls=False,
    ):
        super().__init__()
        self.use_cls = use_cls

        self.linear_layer = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.positional_encoding = PositionalEncoding(
            d_model, max_len=sequence_length
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.linear_layer.weight)
        if self.linear_layer.bias is not None:
            nn.init.zeros_(self.linear_layer.bias)
        if self.use_cls:
            nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, x):
        """
        x: [B, T, input_dim] - input time series
        """
        B, T, _ = x.size()
        # PyTorch transformer expects: True = IGNORE, False = ATTEND
        padding_mask = torch.isnan(x).any(dim=-1)  # [B, T]
        x = torch.nan_to_num(x, nan=0.0)

        x = self.linear_layer(x)  # [B, T, d_model]
        x = self.positional_encoding(x)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        return self.transformer(x, src_key_padding_mask=padding_mask)


class MambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model,
        num_layers,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        self.mamba_layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                )
                for _ in range(num_layers)
            ]
        )

        self.d_model = d_model

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)

        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        returns: [B, T, d_model]
        """

        padding_mask = torch.isnan(x).any(dim=-1)

        x = torch.nan_to_num(
            x,
            nan=0.0,
        )

        x = self.input_projection(x)

        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        for mamba in self.mamba_layers:
            x = mamba(x)

            x = x.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )

        return x


