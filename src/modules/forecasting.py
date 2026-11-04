import torch
import os

import numpy as np
import xarray as xr
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)
from pathlib import Path
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from pytorch_lightning import LightningModule

# from loss import QuantileLoss
from torchmetrics.regression import R2Score


class QuantileLoss(nn.Module):

    def __init__(self, q=0.5):

        super().__init__()

        self.q = q

    def forward(self, pred, target):

        error = target - pred

        return torch.mean(torch.maximum(self.q * error, (self.q - 1) * error))


class ForecastingModule(LightningModule):
    """
    LightningModule for vegetation forecasting.
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.MSELoss()  # QuantileLoss(q=0.5)  #
        self.test_outputs = []

        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if batch is None:
            self.log(
                f"train_loss",
                float("nan"),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        mask = ~torch.isnan(y_true)

        y_pred = y_pred[mask]
        y_true = y_true[mask]

        loss = self.loss_fn(y_pred, y_true)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]

        mask = ~torch.isnan(y_true)

        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]

        loss = self.loss_fn(y_pred_valid, y_true_valid)

        self.val_r2.update(y_pred_valid, y_true_valid)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true_valid.shape[0],
        )

        self.log(
            "val_r2",
            self.val_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None

        y_pred = self(batch)
        y_true = batch["vegetation_forecast"]
        persistence = batch["vegetation_history"][:, -1, :]

        mask = ~torch.isnan(y_true)

        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]

        loss = self.loss_fn(y_pred_valid, y_true_valid)

        self.test_r2.update(y_pred_valid, y_true_valid)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=y_true_valid.shape[0],
        )

        self.log(
            "test_r2",
            self.test_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_outputs.append(
            {
                "preds": y_pred.detach().cpu(),
                "targets": y_true.detach().cpu(),
                "mask": mask.detach().cpu(),
                "forecast_origin": batch["forecast_origin"].cpu(),
                "persistence": persistence.detach().cpu(),
                "sample_id": batch["sample_id"].cpu(),
            }
        )

        return loss

    def on_test_epoch_end(self):

        predictions = (
            torch.cat([x["preds"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        targets = (
            torch.cat([x["targets"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        masks = torch.cat([x["mask"] for x in self.test_outputs]).squeeze(-1).numpy()

        forecast_origins = torch.cat(
            [x["forecast_origin"] for x in self.test_outputs]
        ).numpy()

        persistence = (
            torch.cat([x["persistence"] for x in self.test_outputs]).squeeze(-1).numpy()
        )

        sample_ids = torch.cat([x["sample_id"] for x in self.test_outputs]).numpy()

        dataset = self.trainer.datamodule.test_dataset

        lats = np.array([dataset.locations[int(sid)][0] for sid in sample_ids])

        lons = np.array([dataset.locations[int(sid)][1] for sid in sample_ids])

        ds = xr.Dataset(
            data_vars={
                "predictions": (
                    ("window", "lead_time"),
                    predictions,
                ),
                "targets": (
                    ("window", "lead_time"),
                    targets,
                ),
                "masks": (
                    ("window", "lead_time"),
                    masks,
                ),
                "persistence": (
                    ("window",),
                    persistence,
                ),
            },
            coords={
                "window": np.arange(len(sample_ids)),
                "sample_id": ("window", sample_ids),
                "latitude": ("window", lats),
                "longitude": ("window", lons),
                "forecast_date": (
                    "window",
                    dataset.dataset.time_veg.values[forecast_origins],
                ),
                "lead_time": np.arange(predictions.shape[1]),
            },
        )

        ds.attrs["time_reference"] = "forecast_origin indexes dataset.time_veg"
        ds.attrs["prediction_length"] = np.arange(
            predictions.shape[1]
        )  # self.config.prediction_length

        output_dir = Path(self.config.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        saving_path = output_dir / "predictions.zarr"

        print("Saving path: ", saving_path)
        ds.to_zarr(saving_path, mode="w")

        self.test_outputs.clear()

    def configure_optimizers(self):
        lr = float(self.config.lr)
        warmup_fraction = getattr(self.config, "warmup_fraction", 0)

        total_steps = self.trainer.estimated_stepping_batches
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay,  # L2 norm parameters
        )

        warmup_steps = int(warmup_fraction * total_steps)
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.005,
        )

        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
