import os
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from .datasets import (
    ForecastingTrainDataset,
    ForecastingValDataset,
)

from .batch_sampler import BatchSampler
from .memory_bank import MemoryBank


def days_to_steps(days: int, sampling_days: int) -> int:
    return max(1, round(days / sampling_days))


class ContrastiveDataModule(LightningDataModule):
    """
    DataModule for temporal contrastive learning with a held-out test year.
    """

    def __init__(
        self,
        data_config,
        batch_size=16,
        num_workers=16,
    ):
        """
        test_year: index of the held-out year in Zarr array
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = Path(data_config["path"])
        self.vegetation_indices = data_config["vegetation"]["variables"]
        self.weather_vars = data_config["weather"]["variables"]

        self.train_years = data_config["contrastive"]["train"]
        self.val_years = data_config["contrastive"]["validation"]
        self.test_years = data_config["contrastive"]["test"]
        self.batch_sampler = data_config["contrastive"].get("batch_sampler", False)


    def setup(self, stage=None):
        self.train_dataset = self._build_dataset(years=self.train_years)
        self.val_dataset = self._build_dataset(years=self.val_years)
        self.test_dataset = self._build_dataset(years=self.test_years)

    def _build_dataloader(self, dataset, shuffle=True):

        if self.batch_sampler:
            return DataLoader(
                dataset,
                batch_sampler=BatchSampler(
                    dataset=dataset,
                    shuffle=shuffle,
                ),
                num_workers=self.num_workers,
                collate_fn=safe_collate,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=safe_collate,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False)


class ForecastingDataModule(LightningDataModule):
    def __init__(
        self,
        context_days,
        prediction_days,
        data_config,
        batch_size=16,
        num_workers=16,
    ):
        super().__init__(data_config, batch_size, num_workers)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.context_days = context_days
        self.prediction_days = prediction_days

        self.dataset_path = Path(data_config["path"])
        self.retrieval_path = getattr(data_config, "retrieval_path", None)

        self.train_years = data_config["forecasting"]["train"]
        self.val_years = data_config["forecasting"]["validation"]
        self.test_years = data_config["forecasting"]["test"]

        self.vegetation_indices = data_config["vegetation"]["variables"]
        self.weather_vars = data_config["weather"]["variables"]

        self.sampling_days = data_config["sampling_days"]

        

    def setup(self, stage=None):
        train_dataset = self._build_train_dataset(years=self.train_years)
        val_dataset = self._build_val_dataset(years=self.val_years)
        test_dataset = self._build_val_dataset(years=self.test_years)

        if self.retrieval_path:
            self.train_dataset = MemoryBank(
                train_dataset,
                os.path.join(self.retrieval_path, "train.pt"),
            )

            self.val_dataset = MemoryBank(
                val_dataset,
                os.path.join(self.retrieval_path, "val.pt"),
            )

            self.test_dataset = MemoryBank(
                test_dataset,
                os.path.join(self.retrieval_path, "test.pt"),
            )


    # build_analog_dataset(
    #     dataset=self.train_dataset,
    #     output_path="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/analogs/analog_evi_same_location.pt",
    #     similarity="evi",
    #     k=5,
    #     same_location=True,
    # )
    # build_analog_dataset(
    #     dataset=self.train_dataset,
    #     output_path="/Net/Groups/BGI/tscratch/crobin/ContrastiveEarthnetProject/datasets/analogs/analog_evi_any_location.pt",
    #     similarity="evi",
    #     k=5,
    #     same_location=False,
    # )

    # Build retrieval database ONLY from training data
    # self.analog_database = AnalogDatabase(
    #     dataset=self.train_dataset,
    #     context_length=self.train_dataset.context_length,
    #     prediction_length=self.train_dataset.prediction_length,
    # )

    def _build_train_dataset(self, years):
        return ForecastingTrainDataset(
            dataset_path=self.dataset_path,
            vegetation_indices=self.vegetation_indices,
            weather_vars=self.weather_vars,
            years=years,
            context_days=self.context_days,
            prediction_days=self.prediction_days,
            sampling_days=self.sampling_days,
        )

    def _build_val_dataset(self, years):
        return ForecastingValDataset(
            dataset_path=self.dataset_path,
            vegetation_indices=self.vegetation_indices,
            weather_vars=self.weather_vars,
            years=years,
            context_days=self.context_days,
            prediction_days=self.prediction_days,
            sampling_days=self.sampling_days,
        )

    def _build_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=safe_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset, shuffle=False)


def safe_collate(batch):
    # Remove any None entries
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)
