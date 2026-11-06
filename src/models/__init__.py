from .encoders import TimeSeriesTransformerEncoder, build_encoder

from .forecasters import (
    LSTM,
    LinearRegressionBaseline,
    MLP,
    ForecastModel,
    RAFT,
)


from .benchmark_forcasters import (
    TimeXer,
    TimeMixer,
    PatchTST,
    DLinear,
    ITransformer,
    MICN,
    TimesNet,
    Mamba,
    Transformer,
    TransformerWeatherCovariates,
)


ModelClass = {
    "LSTM": LSTM,
    "LinearRegressionBaseline": LinearRegressionBaseline,
    "MLP": MLP,
    # "TransformerTimeMixer": TransformerTimeMixer,
    "ForecastModel": ForecastModel,
    # External baselines
    "TimeXer": TimeXer,
    "TimeMixer": TimeMixer,
    "PatchTST": PatchTST,
    "DLinear": DLinear,
    "ITransformer": ITransformer,
    "MICN": MICN,
    "TimesNet": TimesNet,
    "Mamba": Mamba,
    "Transformer": Transformer,
    "TransformerWeatherCovariates": TransformerWeatherCovariates,
    "RAFT": RAFT,
}
