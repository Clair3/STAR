import torch
import torch.nn as nn

from .external_loader import load_external_class


OfficialTimeXer = load_external_class(
    repo_name="TimeXer",
    module_path="models/TimeXer.py",
)

OfficialTimeMixer = load_external_class(
    repo_name="TimeXer",
    module_path="models/TimeMixer.py",
)

OfficialPatchTST = load_external_class(
    repo_name="TimeXer",
    module_path="models/PatchTST.py",
)

OfficialDLinear = load_external_class(
    repo_name="TimeXer",
    module_path="models/DLinear.py",
)

OfficialITransformer = load_external_class(
    repo_name="TimeXer",
    module_path="models/iTransformer.py",
)

OfficialMICN = load_external_class(
    repo_name="TimeXer",
    module_path="models/MICN.py",
)

OfficialMamba = load_external_class(
    repo_name="TimeXer",
    module_path="models/Mamba.py",
)

OfficialTimesNet = load_external_class(
    repo_name="TimeXer",
    module_path="models/TimesNet.py",
)

OfficialTransformer = load_external_class(
    repo_name="TimeXer",
    module_path="models/Transformer.py",
)

# ==============================================================
# Generic adapter
# ==============================================================


class ExternalForecastAdapter(nn.Module):
    """
    Adapter between our forecasting dataset and the official
    TimeXer-repository model implementations.

    Common project interface:

        model(data_config, config)
        model(batch) -> [B, prediction_length, 1]

    The external model itself still uses its original interface.
    """

    OFFICIAL_MODEL = None

    def __init__(self, data_config, config):
        super().__init__()

        self.config = config

        # ----------------------------------------------------------
        # Dataset dimensions
        # ----------------------------------------------------------

        self.weather_dim = len(data_config["weather"]["variables"])

        self.target_dim = 1  # EVI

        sampling_days = data_config["sampling_days"]

        self.context_length = round(config.context_days / sampling_days)

        self.prediction_length = round(config.prediction_days / sampling_days)

        self.enc_in = self.weather_dim + self.target_dim

        # ----------------------------------------------------------
        # Build official configuration

        official_config = self._build_config(
            data_config,
            config,
        )

        self.model = self.OFFICIAL_MODEL(official_config)

        # Keep any adapter-specific config values that the
        # official model does not expose as attributes.
        self.official_config = official_config

        # ----------------------------------------------------------
        # Instantiate official model
        # ----------------------------------------------------------

        self.model = self.OFFICIAL_MODEL(official_config)

    # ==========================================================
    # Configuration
    # ==========================================================

    def _build_config(self, data_config, config):
        """
        Construct the configuration object expected by the
        official repository.

        Individual models override this when they require
        additional parameters.
        """

        class C:
            pass

        c = C()

        # ------------------------------------------------------
        # Common forecasting parameters
        # ------------------------------------------------------

        c.task_name = "long_term_forecast"

        c.seq_len = self.context_length
        c.pred_len = self.prediction_length
        c.label_len = getattr(
            config,
            "label_len",
            c.seq_len // 2,
        )
        # c.label_len = 0

        c.enc_in = self.enc_in
        c.c_out = self.enc_in

        c.features = "MS"

        # ------------------------------------------------------
        # Transformer-style parameters
        # ------------------------------------------------------

        c.d_model = getattr(
            config,
            "d_model",
            128,
        )

        c.n_heads = getattr(
            config,
            "num_heads",
            4,
        )

        c.e_layers = getattr(
            config,
            "num_layers",
            2,
        )

        c.d_layers = getattr(config, "d_layers", 1)

        c.d_ff = getattr(
            config,
            "d_ff",
            4 * c.d_model,
        )

        c.dropout = getattr(
            config,
            "dropout",
            0.1,
        )

        c.factor = getattr(
            config,
            "factor",
            5,
        )

        c.activation = getattr(
            config,
            "activation",
            "gelu",
        )

        # ------------------------------------------------------
        # Embedding
        # ------------------------------------------------------

        c.embed = "timeF"
        c.freq = "d"

        # ------------------------------------------------------
        # Normalization
        # ------------------------------------------------------

        c.use_norm = True

        return c

    # ==========================================================
    # Input
    # ==========================================================

    def _build_input(self, batch):
        """
        Convert our batch into the common external-model input.

        Input:

            weather_history: [B, T, W]
            vegetation_history: [B, T, 1]

        Output:

            [B, T, W + 1]

        EVI is deliberately the final channel.
        """

        evi = batch["vegetation_history"]

        weather = batch["weather_history"]

        if evi.ndim != 3:
            raise ValueError(f"Expected EVI [B,T,1], got {evi.shape}")

        if evi.shape[-1] != 1:
            raise ValueError(
                "External benchmark currently expects exactly "
                f"one vegetation target (EVI), got {evi.shape}"
            )

        if weather.ndim != 3:
            raise ValueError(f"Expected weather [B,T,W], got {weather.shape}")

        if weather.shape[0] != evi.shape[0]:
            raise ValueError("Weather and EVI batch dimensions do not match.")

        if weather.shape[1] != evi.shape[1]:
            raise ValueError("Weather and EVI sequence lengths do not match.")

        return torch.cat(
            [
                weather,
                evi,
            ],
            dim=-1,
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, batch):

        x_enc = self._build_input(batch)

        # All models in this repository expose the standard
        # forecasting interface.
        output = self.model(
            x_enc,
            None,
            None,
            None,
        )

        if output is None:
            raise RuntimeError(f"{self.__class__.__name__} returned None.")

        # ------------------------------------------------------
        # Keep EVI target only
        # ------------------------------------------------------

        if output.ndim != 3:
            raise RuntimeError(
                f"{self.__class__.__name__} returned "
                f"shape {output.shape}; expected [B,L,C]."
            )

        # For multivariate models, EVI is our final channel.
        if output.shape[-1] != 1:
            output = output[..., -1:]

        # ------------------------------------------------------
        # Validate forecast horizon
        # ------------------------------------------------------

        if output.shape[1] != self.prediction_length:
            raise RuntimeError(
                f"{self.__class__.__name__} returned forecast "
                f"length {output.shape[1]}, expected "
                f"{self.prediction_length}."
            )

        return output


# ==============================================================
# TimeXer
# ==============================================================


class TimeXer(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialTimeXer

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.patch_len = getattr(
            config,
            "patch_len",
            16,
        )

        return c


# ==============================================================
# TimeMixer
# ==============================================================


class TimeMixer(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialTimeMixer

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        # ------------------------------------------------------
        # TimeMixer-specific configuration
        # ------------------------------------------------------

        c.label_len = getattr(
            config,
            "label_len",
            0,
        )

        c.down_sampling_layers = getattr(
            config,
            "down_sampling_layers",
            2,
        )

        c.down_sampling_window = getattr(
            config,
            "down_sampling_window",
            2,
        )

        c.down_sampling_method = getattr(
            config,
            "down_sampling_method",
            "avg",
        )

        c.channel_independence = getattr(
            config,
            "channel_independence",
            0,
        )

        c.moving_avg = getattr(
            config,
            "moving_avg",
            25,
        )

        c.decomp_method = getattr(
            config,
            "decomp_method",
            "moving_avg",
        )

        c.top_k = getattr(
            config,
            "top_k",
            5,
        )

        c.use_future_temporal_feature = getattr(
            config,
            "use_future_temporal_feature",
            False,
        )

        return c


# ==============================================================
# PatchTST
# ==============================================================


class PatchTST(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialPatchTST

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.patch_len = getattr(
            config,
            "patch_len",
            16,
        )

        c.stride = getattr(
            config,
            "stride",
            8,
        )

        return c


# ==============================================================
# DLinear
# ==============================================================


class DLinear(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialDLinear

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.moving_avg = getattr(
            config,
            "moving_avg",
            25,
        )

        c.individual = getattr(
            config,
            "individual",
            False,
        )

        return c


# ==============================================================
# iTransformer
# ==============================================================


class ITransformer(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialITransformer

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        return c


class MICN(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialMICN

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.decomp_kernel = getattr(
            config,
            "micn_decomp_kernel",
            25,
        )

        c.dec_in = c.enc_in
        c.c_out = 1

        c.d_layers = getattr(
            config,
            "d_layers",
            1,
        )

        c.label_len = getattr(
            config,
            "label_len",
            min(18, c.seq_len),
        )

        return c

    def forward(self, batch):

        x_enc = self._build_input(batch)

        B = x_enc.shape[0]
        pred_len = self.model.pred_len
        label_len = self.official_config.label_len

        # Historical decoder context only.
        dec_context = x_enc[:, -label_len:, :]

        # Unknown future values are represented by zeros.
        dec_future = torch.zeros(
            B,
            pred_len,
            x_enc.shape[-1],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )

        x_dec = torch.cat(
            [
                dec_context,
                dec_future,
            ],
            dim=1,
        )

        output = self.model(
            x_enc,
            None,
            x_dec,
            None,
        )

        if output.ndim != 3:
            raise RuntimeError(
                f"MICN returned shape {output.shape}; " "expected [B,L,C]."
            )

        return output[..., -1:]


class Mamba(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialMamba

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        # IMPORTANT:
        # The official TimeXer Mamba implementation uses
        # configs.d_ff as Mamba's d_state.
        c.d_ff = getattr(
            config,
            "mamba_d_state",
            64,
        )

        c.d_conv = getattr(
            config,
            "mamba_d_conv",
            4,
        )

        c.expand = getattr(
            config,
            "mamba_expand",
            2,
        )

        return c


class TimesNet(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialTimesNet

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        # TimesNet-specific parameters
        c.top_k = getattr(
            config,
            "timesnet_top_k",
            5,
        )

        c.num_kernels = getattr(
            config,
            "timesnet_num_kernels",
            6,
        )

        return c


class Transformer(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialTransformer

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.dec_in = c.enc_in
        c.c_out = 1

        c.d_layers = getattr(
            config,
            "d_layers",
            1,
        )

        c.label_len = getattr(
            config,
            "label_len",
            min(18, c.seq_len),
        )

        return c

    def forward(self, batch):

        evi = batch["vegetation_history"]
        weather = batch["weather_history"]

        x_enc = torch.cat(
            [
                weather,
                evi,
            ],
            dim=-1,
        )

        B = x_enc.shape[0]
        pred_len = self.model.pred_len
        label_len = self.official_config.label_len

        dec_context = x_enc[:, -label_len:, :]

        dec_future = torch.zeros(
            B,
            pred_len,
            x_enc.shape[-1],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )

        x_dec = torch.cat(
            [
                dec_context,
                dec_future,
            ],
            dim=1,
        )

        output = self.model(
            x_enc,
            None,
            x_dec,
            None,
        )

        return output[..., -1:]


class TransformerWeatherCovariates(ExternalForecastAdapter):

    OFFICIAL_MODEL = OfficialTransformer

    def _build_config(self, data_config, config):

        c = super()._build_config(
            data_config,
            config,
        )

        c.dec_in = c.enc_in
        c.c_out = 1

        c.d_layers = getattr(
            config,
            "d_layers",
            1,
        )

        c.label_len = getattr(
            config,
            "label_len",
            min(18, c.seq_len),
        )

        return c

    def forward(self, batch):

        vegetation_history = batch["vegetation_history"]
        weather_history = batch["weather_history"]
        weather_forecast = batch["weather_forecast"]

        x_enc = torch.cat(
            [
                weather_history,
                vegetation_history,
            ],
            dim=-1,
        )

        # Historical decoder context
        label_len = self.official_config.label_len

        dec_context = torch.cat(
            [
                weather_history[:, -label_len:],
                vegetation_history[:, -label_len:],
            ],
            dim=-1,
        )

        # Future:
        # weather is known
        # EVI is unknown → zeros
        vegetation_forecast = torch.zeros(
            weather_forecast.shape[0],
            weather_forecast.shape[1],
            1,
            device=weather_forecast.device,
            dtype=weather_forecast.dtype,
        )

        dec_future = torch.cat(
            [
                weather_forecast,
                vegetation_forecast,
            ],
            dim=-1,
        )

        x_dec = torch.cat(
            [
                dec_context,
                dec_future,
            ],
            dim=1,
        )

        output = self.model(
            x_enc,
            None,
            x_dec,
            None,
        )

        return output[..., -1:]
