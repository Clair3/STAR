from datetime import datetime
import os
import torch
import wandb
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from contextlib import contextmanager

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from data import ContrastiveDataModule, ForecastingDataModule
from modules import ContrastiveModule, ForecastingModule
from models import (
    TimeSeriesTransformerEncoder,
    CLSHead,
    ModelClass,
)

PROJECT_DIR = Path(__file__).parent.parent.resolve()
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BaseExperiment:
    def __init__(self, config, data_config):
        if isinstance(config, dict):
            config = SimpleNamespace(**config)
        self.config = config
        self.data_config = data_config
        self.logger = WandbLogger(project="contrastive-earthnet", log_model="best")

    def build_datamodule(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def build_callbacks(self, run_id):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(CHECKPOINT_DIR, run_id),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            save_weights_only=False,  # ensures full reproducibility
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=25,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        return [checkpoint, early_stop, lr_monitor]

    def build_trainer(self, logger=None):
        return Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=self.config.gpu_device,
            val_check_interval=0.2,
            precision="16-mixed",
            logger=logger,
            callbacks=self.build_callbacks(wandb.run.id) if wandb.run else None,
            log_every_n_steps=32,
            gradient_clip_val=self.config.gradient_clip_val,
            deterministic=True,
            accumulate_grad_batches=1,
            # (
            #     30 if self.data_config["contrastive"].get("batch_sampler", False) else 1
            # ),
            enable_progress_bar=True,
        )

    def run(self, profile_resources=False):
        datamodule = self.build_datamodule()
        model = self.build_model()

        if profile_resources:
            print("Running dry-run on single batch to estimate resources...")
            self.estimate_batch_resources(model, datamodule)
            return  # skip full training

        trainer = self.build_trainer(logger=self.logger)
        trainer.fit(model, datamodule)

        ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.test(model=None, datamodule=datamodule, ckpt_path=ckpt_path)

    def load_model(self, model, run_name):
        ckpt_path = CHECKPOINT_DIR / run_name / "best.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        return model

    def eval(self, run_name):
        datamodule = self.build_datamodule()
        model = self.build_model()
        model = self.load_model(model, run_name)

        trainer = self.build_trainer(logger=False)

        trainer.test(model=model, datamodule=datamodule)

    @staticmethod
    def estimate_batch_resources(model, datamodule, device="cuda"):
        """
        Run a single batch through the model to check memory and speed.
        """
        model = model.to(device)
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = model(batch) if callable(model) else model.forward(batch)
        if hasattr(output, "sum"):  # make a dummy loss
            loss = output.sum()
            loss.backward()
        end.record()
        torch.cuda.synchronize()

        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Single-batch GPU memory used: {mem_peak:.1f} MB")
        print(f"Time per batch: {start.elapsed_time(end)/1000:.3f} s")


class ContrastiveExperiment(BaseExperiment):

    def build_datamodule(self):
        return ContrastiveDataModule(
            data_config=self.data_config,
            batch_size=self.config.batch_size,
            num_workers=self.config["num_workers"],
        )

    def build_model(self):
        encoder_veg = TimeSeriesTransformerEncoder(
            input_dim=len(self.data_config["vegetation"]["variables"]),
            sequence_length=self.data_config["vegetation"]["sequence_length"],
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_cls=self.config.use_cls,
            seasonal_positional_encoding=self.config.seasonal_positional_encoding,
        )

        encoder_weather = TimeSeriesTransformerEncoder(
            input_dim=len(self.data_config["weather"]["variables"]),
            sequence_length=self.data_config["weather"]["sequence_length"],
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_cls=self.config.use_cls,
            seasonal_positional_encoding=self.config.seasonal_positional_encoding,
        )

        self.encoder_veg = encoder_veg
        self.encoder_weather = encoder_weather

        vegetation_model = CLSHead(encoder_veg)
        weather_model = CLSHead(encoder_weather)

        contrastive_module = ContrastiveModule(
            encoder_veg=vegetation_model,
            encoder_weather=weather_model,
            cfg=dict(self.config),
        )
        return contrastive_module


class ForecastingExperiment(BaseExperiment):

    def build_datamodule(self):
        return ForecastingDataModule(
            context_days=self.config.context_days,
            prediction_days=self.config.prediction_days,
            data_config=self.data_config,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def build_model(self):
        # if self.config.model_name == "TransformerBaseline":
        #
        #     self.model = ModelClass[self.config.model_name](
        #         data_config=self.data_config,
        #         config=self.config,
        #     )
        #     pretrained_checkpoint = getattr(self, "pretrained_checkpoint", None)
        #     self.model= self.load_model(self.model, run_name)
        # else:
        self.model = ModelClass[self.config.model_name](
            data_config=self.data_config,
            config=self.config,
        )

        forecasting_module = ForecastingModule(
            model=self.model,
            config=self.config,
        )
        return forecasting_module


class PretrainThenForecastExperiment(BaseExperiment):
    """
    Runs contrastive pretraining first, then uses the pretrained encoders
    for forecasting.
    """

    def __init__(self, config, data_config):
        super().__init__(config, data_config)

        self.contrastive_experiment = ContrastiveExperiment(config, data_config)
        self.forecasting_experiment = ForecastingExperiment(config, data_config)

    def run(self, profile_resources=False):
        # Step 1: Pretrain contrastive model
        print("=== Step 1: Contrastive pretraining ===")
        self.contrastive_experiment.run(profile_resources=profile_resources)

        # Load pretrained encoders
        encoder_veg = self.contrastive_experiment.encoder_veg
        encoder_weather = self.contrastive_experiment.encoder_weather

        # Step 2: Forecasting with pretrained encoders
        print("=== Step 2: Forecasting finetuning ===")
        self.forecasting_experiment.pretrained_encoders = {
            "veg": encoder_veg,
            "weather": encoder_weather,
        }
        self.forecasting_experiment.run(profile_resources=profile_resources)


@contextmanager
def wandb_run(config, group=None, project="contrastive-earthnet", run_name=None):
    if run_name is not None:
        api = wandb.Api()
        run = api.run(f"paper_2/{project}/{run_name}")
        yield run.config

    else:
        run = wandb.init(project=project, group=group, config=config)
        try:
            yield run.config
        finally:
            run.finish()


def run_experiment(config, data_config, profile_resources=False, run_name=None):
    task = config["task"]

    if task == "contrastive":
        experiment = ContrastiveExperiment(config, data_config)
    elif task == "forecasting":
        experiment = ForecastingExperiment(config, data_config)
    elif task == "pretrain_then_forecast":
        experiment = PretrainThenForecastExperiment(config, data_config)
    else:
        raise ValueError(f"Unknown experiment task: {task}")
    if run_name is not None:
        experiment.eval(run_name=run_name)
    else:
        experiment.run(profile_resources=profile_resources)


def get_folds(data_config, task, kfolds=True):
    split = data_config[task]
    if kfolds:
        return [(train, test, test) for train, test in data_config["kfolds"]]
    else:
        return [(split["train"], split.get("validation", split["test"]), split["test"])]


def run_pipeline(
    train_config_file="models/mlp.yaml",
    data_config_file="data_config.yaml",
    experiment_name="",
    run_name=None,
    profile_resources=False,
):
    """
    Run training and evaluation experiments in different execution modes.

    This is the main entry point for launching experiments. It supports:
    - single-run training
    - hyperparameter tuning on one split
    - k-fold cross-validation with fixed hyperparameters

    Parameters
    ----------
    train_config_file : str
        Path (relative to CONFIG_DIR) to the training config (e.g., "models/mlp.yaml").
        Must include a "task" field (e.g., "contrastive", "forecasting").
        Defines model architecture, training settings, and optional sweep config.

    data_config_file : str
        Path (relative to CONFIG_DIR) to the data config (e.g., "data_config.yaml").

    mode : {"single", "tune", "kfold"}, default="single"
        Execution mode:

        - "single": standard train/val/test split from data config
        - "tune": same as "single", but outputs stored under "tuning/" (for sweeps/debugging)
        - "kfold": runs across predefined folds with fixed hyperparameters

    folds : list of tuple(list[int], list[int]), optional
        Required for "kfold". Each tuple is (train_years, test_years), e.g.:
            [([2017, 2018], [2019]), ([2017, 2018, 2019], [2020])]
        Ignored otherwise.

    profile_resources : bool, default=False
        If True, runs a single batch to estimate resource usage.

    Behavior
    --------
    - Load configs and set a global seed.
    - Iterate over folds based on mode.
    - For each fold:
        - update data splits
        - create output directory
        - start a wandb run
        - execute training/evaluation

    Outputs
    -------
    Stored under:

        PREDICTIONS_DIR / <model_name> /

    Subdirectories:
        - "single":  single/
        - "tune":    tuning/
        - "kfold":   fold_<test_years>/

    Notes
    -----
    - "tune" uses a single fold but isolates outputs.
    - "kfold" reuses the same config across folds.
    - Assumes valid configs with required fields.
    """

    print(
        f"Loading training config from {train_config_file} and data config from {data_config_file}..."
    )
    data_config = load_config(CONFIG_DIR / data_config_file)
    train_config = load_config(CONFIG_DIR / train_config_file)

    seed_everything(42, workers=True)

    base_output_dir = (
        PREDICTIONS_DIR
        / data_config["vegetation"]["sensor"]
        / train_config["model_name"]
    )

    train_config["output_dir"] = (
        base_output_dir
        / experiment_name
        / Path(str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        # / str(data_config["forecasting"]["test"][-1])
    )
    os.makedirs(train_config["output_dir"], exist_ok=True)

    fold_config = deepcopy(train_config)

    with wandb_run(
        fold_config, group=fold_config["model_name"], run_name=run_name
    ) as config:
        run_experiment(
            config,
            data_config,
            run_name=run_name,
            profile_resources=profile_resources,
        )

    print("All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_config_file",
        default="defaults/transformer_timemixer.yaml",  # _pretrain_forecast.yaml",
        help="Path to training config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--data_config_file",
        default="data_config_VIIRS_passage.yaml",
        help="Path to data config file (relative to project/configs/)",
    )
    parser.add_argument(
        "--run_name",
        default=None, #"fdq12lyf", # "9w14zh8e",  # None, #"d7s0pwof",  # None,  # "1id511n8",  # "40g3alip",  # ng4jsfgi",  # None,  # "bderu1i9",  # "bderu1i9", "scd7t68g", # None #"aj40n2mg", # None, #"0vh39sji", #"5umw093g",  # None,  # "chkpceo7", #"ewdpengg",  # None,  # "yom83w3e",  # None,  # "ho47mfll",  # None,  # "rysja5hl",  # None, # "lxqlhali",  # None, #"6tvil6wq",  # "r30aeonh",  # None,  # "6tvil6wq",  # None,  # "o03jeetw", # None, #"j3fgpisl", #"za90ubzz",  # None,  # "ug77kly8",  # None,  # "yqbclumu",  # None,  # "d0686mzv",  # None,  # "ppj1fh2v", #None,  # "a28nl7qz",  # None,  # "fleagmb2",  # None,  # "g4u7970n",  # "k6qige6q",  # "qoj0owg6",  # "53mphdie",  # "j1oq2t6k",  # "0epb2ml8",  # "j1oq2t6k", "qoj0owg6",  #
        help="Path of the model weights. If None, the training of the experiment is executed. If a path is provided, the evaluation mode is executed.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "tune", "kfold"],
        default="single",
        help="Execution mode: 'single' for standard train/val/test split, 'tune' for hyperparameter tuning on a single fold, 'kfold' for k-fold cross-validation with predefined folds",
    )

    parser.add_argument(
        "--experiment_name",
        default="NewModels",
        help="name of the folder used to save the test set (outputs/predictions/dataset/model/experiment_name/time)",
    )

    parser.add_argument(
        "--profile_resources",
        action="store_true",
        help="Run a single batch for resource estimation",
    )

    args = parser.parse_args()

    print(f"args are: {args}")

    run_pipeline(
        train_config_file=args.train_config_file,
        data_config_file=args.data_config_file,
        experiment_name=args.experiment_name,
        profile_resources=args.profile_resources,
        run_name=args.run_name,
    )
