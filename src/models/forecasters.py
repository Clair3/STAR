import torch
import torch.nn as nn

import torch.nn.functional as F
import copy
import math
import torch
import torch.nn as nn

from data.datasets import days_to_steps
from .encoders import TimeSeriesTransformerEncoder, build_encoder


class LinearRegressionBaseline(nn.Module):
    def __init__(self, data_config, config=None):

        super().__init__()
        self.veg_dim = len(data_config["vegetation"]["variables"])
        self.weather_dim = len(data_config["weather"]["variables"])
        self.veg_seq_len = data_config["vegetation"]["sequence_length"]
        self.weather_seq_len = data_config["weather"]["sequence_length"]
        input_dim = (
            self.veg_seq_len * self.veg_dim
            + 2 * self.weather_seq_len * self.weather_dim
        )
        self.linear = nn.Linear(input_dim, self.veg_seq_len * self.veg_dim)

    def forward(self, batch):
        veg = batch["vegetation_history"]
        weather = batch["weather_history"]
        forecast = batch["weather_forecast"]

        # Replace NaNs with zero
        veg = torch.nan_to_num(veg, nan=0.0)
        weather = torch.nan_to_num(weather, nan=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0)

        x = torch.cat(
            [veg.flatten(1), weather.flatten(1), forecast.flatten(1)],
            dim=-1,
        )
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, data_config, config=None):

        super().__init__()
        self.veg_dim = len(data_config["vegetation"]["variables"])
        self.weather_dim = len(data_config["weather"]["variables"])
        self.veg_seq_len = data_config["vegetation"]["sequence_length"]
        self.weather_seq_len = data_config["weather"]["sequence_length"]

        hidden_dim = config.hidden_dim if config is not None else 128

        input_dim = (
            self.veg_seq_len * self.veg_dim
            + 2 * self.weather_seq_len * self.weather_dim
        )
        output_dim = self.veg_seq_len * self.veg_dim
        # print(input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        veg_hist = batch["vegetation_history"]  # [B, T_veg, C_veg]
        weather = batch["weather_history"]  # [B, T_w, C_w]
        forecast = batch["weather_forecast"]  # [B, T_w, C_w]

        smoothed = F.avg_pool1d(
            torch.nan_to_num(veg_hist, nan=0.0).transpose(1, 2),
            kernel_size=5,
            stride=1,
            padding=5 // 2,
        ).transpose(1, 2)

        # Only replace NaNs, keep original values intact
        veg_hist = torch.where(torch.isnan(veg_hist), smoothed, veg_hist)

        # Replace NaNs with zero
        veg = torch.nan_to_num(veg_hist, nan=0.0)
        weather = torch.nan_to_num(weather, nan=0.0)
        forecast = torch.nan_to_num(forecast, nan=0.0)

        x = torch.cat(
            [veg.flatten(1), weather.flatten(1), forecast.flatten(1)],
            dim=-1,
        )
        out = self.mlp(x)
        if out.ndim == 2:  # [B, T]
            out = out.unsqueeze(-1)
        return out


class LSTM(nn.Module):
    def __init__(self, data_config, config=None):
        super().__init__()
        veg_dim = len(data_config["vegetation"]["variables"])
        weather_dim = len(data_config["weather"]["variables"])

        input_dim = veg_dim + weather_dim
        hidden_dim = config.hidden_dim if config is not None else 128
        num_layers = config.num_layers if config is not None else 1
        dropout = config.dropout if config is not None else 0.0
        print(
            f"Initializing LSTM with hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}"
        )
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Prediction head: maps hidden state to vegetation
        self.head = nn.Linear(hidden_dim, veg_dim)

    def teacher_forcing_ratio(self, step=None, total_steps=None):
        # Compute teacher forcing probability using a sigmoid schedule
        # if step is not None and total_steps is not None:
        #     # Start mostly teacher forcing, decay over time
        #     return 1 / (1 + torch.exp(5 * (step / total_steps - 0.5)))  # sigmoid
        # else:
        return 0.5

    def forward(self, batch, step=None, total_steps=None):
        """
        step: current training step (int)
        total_steps: total number of training steps (int)
        The teacher forcing ratio will be computed with a sigmoid schedule.
        """
        veg_hist = batch["vegetation_history"]
        weather_hist = batch["weather_history"]
        weather_forecast = batch["weather_forecast"]

        # Smooth NaNs in vegetation history
        smoothed = F.avg_pool1d(
            torch.nan_to_num(veg_hist, nan=0.0).transpose(1, 2),
            kernel_size=5,
            stride=1,
            padding=5 // 2,
        ).transpose(1, 2)

        # Only replace NaNs, keep original values intact
        veg_hist = torch.where(torch.isnan(veg_hist), smoothed, veg_hist)
        veg_hist = torch.where(torch.isnan(veg_hist), 0, veg_hist)

        if self.training:
            veg_forecast = batch["vegetation_forecast"]

        B, T_future, _ = weather_forecast.shape

        # History encoding
        x_hist = torch.cat([veg_hist, weather_hist], dim=-1)
        _, (h, c) = self.lstm(x_hist)

        input_veg = veg_hist[:, -1, :]
        preds = []

        for t in range(T_future):
            weather_step = weather_forecast[:, t, :]
            lstm_input = torch.cat([input_veg, weather_step], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            # Predict only the difference from previous step
            pred_step = input_veg + self.head(out.squeeze(1))
            preds.append(pred_step.unsqueeze(1))

            # Decide teacher forcing per batch element
            if self.training:
                tf_prob = self.teacher_forcing_ratio(step=step, total_steps=total_steps)
                mask = (torch.rand(B, device=pred_step.device) < tf_prob).unsqueeze(1)

                # Only use non-NaN values for teacher forcing
                veg_step = veg_forecast[:, t, :]

                # valid teacher forcing positions
                valid_tf = mask & (~torch.isnan(veg_step))

                input_veg = torch.where(valid_tf, veg_step, pred_step)
            else:
                input_veg = pred_step  # no teacher forcing at inference
        return torch.cat(preds, dim=1)



class DayOfYearEncoding(nn.Module):
    """Stateless: turns integer day-of-year (0-364) into 2 channels (sin, cos)."""

    def __init__(self, period: int = 365):
        super().__init__()
        self.period = period

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        # doy: [B, T] integer or float day-of-year
        angle = 2 * math.pi * doy.float() / self.period
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # [B, T, 2]


# ----------------------------------------------------------------------
# Simple TimeMixer block
# ----------------------------------------------------------------------
class MixerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.time_norm = nn.LayerNorm(d_model)
        self.feature_norm = nn.LayerNorm(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        # TODO: maybe something weird here to check
        # temporal mixing
        y = self.time_norm(x)
        y = y.transpose(1, 2)
        y = self.time_mlp(y.transpose(1, 2))
        x = x + y

        # feature mixing
        y = self.feature_norm(x)
        y = self.feature_mlp(y)
        x = x + y

        return x


class ForecastModel(nn.Module):

    def __init__(self, data_config, config):

        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        # --------------------------------------------------------------
        # Day-of-year encoding
        # --------------------------------------------------------------

        self.doy_encoding = DayOfYearEncoding()

        encoder_kwargs = {
            "num_layers": config.num_layers,
        }

        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)

        if config.encoder_name == "transformer":
            encoder_kwargs.update(
                {
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,
                }
            )

        elif config.encoder_name == "mamba":
            encoder_kwargs.update(
                {
                    "mamba_d_state": config.mamba_d_state,
                    "mamba_d_conv": config.mamba_d_conv,
                    "mamba_expand": config.mamba_expand,
                    "mamba_headdim": config.mamba_headdim,
                }
            )
        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)
        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------

        self.hist_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=climate_dim + 2,
            sequence_length=context_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Future weather
        # --------------------------------------------------------------

        self.weather_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=weather_dim + 2,
            sequence_length=prediction_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Cross-attention
        # --------------------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(
                2 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --------------------------------------------------------------
        # Mixer
        # --------------------------------------------------------------

        self.mixers = nn.Sequential(
            *[
                MixerBlock(
                    d_model=d_model,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # --------------------------------------------------------------
        # Prediction head
        # --------------------------------------------------------------

        self.head = nn.Linear(
            d_model,
            veg_dim,
        )

    def _with_doy(self, x, doy):

        return torch.cat(
            [
                x,
                self.doy_encoding(doy),
            ],
            dim=-1,
        )

    def forward(self, batch):

        # ==============================================================
        # History
        # ==============================================================

        vegetation_history = batch["vegetation_history"]

        weather_history = batch["weather_history"]

        # --------------------------------------------------------------
        # Concatenate normalized history
        # --------------------------------------------------------------

        hist = torch.cat(
            [
                vegetation_history,
                weather_history,
            ],
            dim=-1,
        )

        hist = self._with_doy(
            hist,
            batch["hist_doy"],
        )

        hist_features = self.hist_encoder(hist)

        # ==============================================================
        # Future weather
        # ==============================================================

        weather = batch["weather_forecast"]

        weather = self._with_doy(
            weather,
            batch["forecast_doy"],
        )

        weather_features = self.weather_encoder(weather)

        # ==============================================================
        # Retrieve relevant history
        # ==============================================================

        history_context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
        )

        # ==============================================================
        # Fusion
        # ==============================================================

        x = torch.cat(
            [
                weather_features,
                history_context,
                # aligned_climate,
            ],
            dim=-1,
        )

        x = self.fusion(x)

        # ==============================================================
        # Mixer
        # ==============================================================

        x = self.mixers(x)

        # ==============================================================
        # Forecast in normalized vegetation space
        # ==============================================================

        prediction = self.head(x)

        return prediction


class RetrievalModel(nn.Module):

    def __init__(self, data_config, config):

        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        weather_dim = len(data_config["weather"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        climate_dim = veg_dim + weather_dim

        d_model = config.d_model

        # --------------------------------------------------------------
        # Day-of-year encoding
        # --------------------------------------------------------------

        self.doy_encoding = DayOfYearEncoding()

        encoder_kwargs = {
            "num_layers": config.num_layers,
        }

        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)

        if config.encoder_name == "transformer":
            encoder_kwargs.update(
                {
                    "num_heads": config.num_heads,
                    "dropout": config.dropout,
                }
            )

        elif config.encoder_name == "mamba":
            encoder_kwargs.update(
                {
                    "mamba_d_state": config.mamba_d_state,
                    "mamba_d_conv": config.mamba_d_conv,
                    "mamba_expand": config.mamba_expand,
                    "mamba_headdim": config.mamba_headdim,
                }
            )
        print("encoder_name:", config.encoder_name)
        print("encoder_kwargs:", encoder_kwargs)



        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------

        self.hist_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=climate_dim + 2,
            sequence_length=context_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Future weather
        # --------------------------------------------------------------

        self.weather_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=weather_dim + 2,
            sequence_length=prediction_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Retrieval encoder
        # --------------------------------------------------------------

        self.retrieval_encoder = build_encoder(
            name=config.encoder_name,
            input_dim=climate_dim + 2,
            sequence_length=prediction_length,
            d_model=d_model,
            **encoder_kwargs,
        )

        # --------------------------------------------------------------
        # Cross-attention
        # --------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --------------------------------------------------------------
        # Mixer
        # --------------------------------------------------------------
        self.mixers = nn.Sequential(
            *[
                MixerBlock(
                    d_model=d_model,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # --------------------------------------------------------------
        # Prediction head
        # --------------------------------------------------------------

        self.head = nn.Linear(
            d_model,
            veg_dim,
        )

    def _with_doy(self, x, doy):

        return torch.cat(
            [
                x,
                self.doy_encoding(doy),
            ],
            dim=-1,
        )

    def forward(self, batch):

        # ==============================================================
        # History
        # ==============================================================

        retrievals = batch["retrievals"]


        vegetation_history = batch["vegetation_history"]

        weather_history = batch["weather_history"]

        # --------------------------------------------------------------
        # Concatenate normalized history
        # --------------------------------------------------------------

        hist = torch.cat(
            [
                vegetation_history,
                weather_history,
            ],
            dim=-1,
        )

        hist = self._with_doy(
            hist,
            batch["hist_doy"],
        )

        hist_features = self.hist_encoder(hist)

        # ==============================================================
        # Future weather
        # ==============================================================

        weather = batch["weather_forecast"]

        weather = self._with_doy(
            weather,
            batch["forecast_doy"],
        )

        weather_features = self.weather_encoder(weather)

        # ==============================================================
        # Retrieve relevant history
        # ==============================================================

        history_context, _ = self.cross_attention(
            query=weather_features,
            key=hist_features,
            value=hist_features,
        )

        # ==============================================================
        # Fusion
        # ==============================================================

        x = torch.cat(
            [
                weather_features,
                history_context,
            ],
            dim=-1,
        )

        x = self.fusion(x)

        # ==============================================================
        # Mixer
        # ==============================================================

        x = self.mixers(x)

        # ==============================================================
        # Forecast in normalized vegetation space
        # ==============================================================

        prediction = self.head(x)

        return prediction



class RAFT(nn.Module):
    """
    RAFT baseline adapted to the project's ForecastingDataModule interface.

    Input:
        batch["vegetation_history"]:
            [B, context_length, vegetation_dim]

        batch["retrieval"]:
            [B, topm, prediction_length, vegetation_dim]

        batch["retrieval_weights"]:
            [B, topm]

    Output:
        [B, prediction_length, vegetation_dim]
    """

    def __init__(self, data_config, config):
        super().__init__()

        veg_dim = len(data_config["vegetation"]["variables"])

        context_length = days_to_steps(
            config.context_days,
            data_config["sampling_days"],
        )

        prediction_length = days_to_steps(
            config.prediction_days,
            data_config["sampling_days"],
        )

        self.veg_dim = veg_dim
        self.context_length = context_length
        self.prediction_length = prediction_length

        # --------------------------------------------------
        # Direct forecasting branch
        # --------------------------------------------------

        self.linear_x = nn.Linear(
            context_length,
            prediction_length,
        )

        # --------------------------------------------------
        # Retrieval branch
        #
        # The retrieved future already has the correct
        # prediction length, but RAFT learns a transformation
        # before combining it with the direct forecast.
        # --------------------------------------------------

        self.linear_retrieval = nn.Linear(
            prediction_length,
            prediction_length,
        )

        # --------------------------------------------------
        # Combine direct + retrieval forecasts
        # --------------------------------------------------

        self.linear_pred = nn.Linear(
            2 * prediction_length,
            prediction_length,
        )

    def forward(self, batch):

        # ==================================================
        # Input
        # ==================================================

        x = batch["vegetation_history"]

        # [B, context_length, veg_dim]

        retrieval = batch["retrieval"]

        # [B, topm, prediction_length, veg_dim]

        weights = batch["retrieval_weights"]

        # [B, topm]

        # ==================================================
        # Direct forecasting branch
        # ==================================================

        # Last observed EVI value.
        #
        # This is the same offset-style treatment used
        # in the original RAFT implementation.

        x_offset = x[:, -1:, :].detach()

        x_norm = x - x_offset

        # [B, context_length, C]
        #       ↓
        # [B, C, context_length]
        #       ↓
        # [B, C, prediction_length]
        #       ↓
        # [B, prediction_length, C]

        x_pred = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        # ==================================================
        # Retrieval branch
        # ==================================================

        # Weighted combination of the top-k retrieved
        # future trajectories.
        #
        # retrieval:
        #     [B, K, P, C]
        #
        # weights:
        #     [B, K]
        #
        # result:
        #     [B, P, C]

        retrieval_pred = (retrieval * weights[:, :, None, None]).sum(dim=1)

        # ==================================================
        # Retrieval transformation
        # ==================================================

        retrieval_pred = self.linear_retrieval(retrieval_pred.permute(0, 2, 1)).permute(
            0, 2, 1
        )

        # ==================================================
        # Combine
        # ==================================================

        # [B, P, C] + [B, P, C]
        #
        # concatenate along the temporal dimension
        # because the original RAFT implementation applies
        # its final linear layer channel-wise.

        pred = torch.cat(
            [
                x_pred,
                retrieval_pred,
            ],
            dim=1,
        )

        # [B, 2P, C]
        #      ↓
        # [B, C, 2P]
        #      ↓
        # [B, C, P]
        #      ↓
        # [B, P, C]

        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)

        # Restore original EVI level.

        pred = pred + x_offset

        return pred
