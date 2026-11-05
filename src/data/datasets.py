import xarray as xr
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from torchvision.transforms import Compose

from .transform import RandomFeatureMask, RandomGaussianJitter


def days_to_steps(days: int, sampling_days: int) -> int:
    return max(1, round(days / sampling_days))


class BaseDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        vegetation_indices,
        weather_vars,
        years,
        context_days,
        prediction_days,
        sampling_days=1,
    ):
        self.vegetation_indices = vegetation_indices
        self.weather_vars = weather_vars

        self.dataset = self.dataset_split(
            dataset_path=dataset_path,
            years=years,
            context_days=context_days,
        )

        self.num_samples = len(self.dataset.sample)
        self.sampling_days = sampling_days

        self.context_length = days_to_steps(
            context_days,
            sampling_days,
        )

        self.prediction_length = days_to_steps(
            prediction_days,
            sampling_days,
        )

        self.observations_per_year = days_to_steps(
            365,
            sampling_days,
        )

        print(f"context: {context_days} days -> {self.context_length} steps")
        print(f"prediction: {prediction_days} days -> {self.prediction_length} steps")

        self.stride = days_to_steps(5, sampling_days)

        self._create_training_pairs()
        self._preload_data()

        self.locations = {
            sample_id: (
                float(self.dataset.latitude.values[sample_id]),
                float(self.dataset.longitude.values[sample_id]),
            )
            for sample_id in range(self.num_samples)
        }

    def dataset_split(
        self,
        dataset_path: str,
        years: list[int],
        context_days: int,
    ) -> xr.Dataset:
        """
        Create a dataset split containing the requested years plus
        enough preceding data to provide the temporal context.
        """
        dataset = xr.open_zarr(dataset_path)

        print(
            f"Loading dataset from {dataset_path} "
            f"and filtering for years {years} "
            f"with {context_days} days of context"
        )

        start_date = pd.Timestamp(f"{min(years)}-01-01") - pd.Timedelta(
            days=context_days
        )
        end_date = pd.Timestamp(f"{max(years)}-12-31 23:59:59")

        return dataset.sel(
            time_veg=slice(start_date, end_date),
            time_weather=slice(start_date, end_date),
        ).load()

    def _create_training_pairs(self):

        n_time = len(self.dataset.time_veg)

        origins = range(
            self.context_length,
            n_time - self.prediction_length + 1,
            self.stride,
        )
        self.training_pairs = [
            (sample_id, t0) for sample_id in range(self.num_samples) for t0 in origins
        ]

        print(
            f"n_time={n_time}, "
            f"context={self.context_length}, "
            f"prediction={self.prediction_length}"
        )

        print(
            f"num_samples={self.num_samples}, "
            f"num_windows={len(self.training_pairs)}"
        )

    def _preload_data(self):
        self.vegetation = []
        self.weather = []
        self._build_doy_lookup()
        for sample_id in range(self.num_samples):
            sample = self.dataset.isel(sample=sample_id)
            veg = torch.as_tensor(
                np.stack(
                    [sample[var].values for var in self.vegetation_indices],
                    axis=1,
                ),
                dtype=torch.float32,
            )
            weather = torch.as_tensor(
                np.stack(
                    [sample[var].values for var in self.weather_vars],
                    axis=1,
                ),
                dtype=torch.float32,
            )
            self.vegetation.append(veg)
            self.weather.append(weather)

    def _preload_msc(self, sample_id):
        msc = self.dataset.isel(sample=sample_id).msc
        # if msc.size == 366:
        #     msc = msc.drop_isel(dayofyear=59)  # leap year
        msc_arr = torch.as_tensor(
            msc.values,
            dtype=torch.float32,
        ).unsqueeze(-1)
        return msc_arr

    def _preload_weather_msc(self, sample_id):

        sample = self.dataset.isel(sample=sample_id)

        weather_msc = np.stack(
            [sample[f"{var}_msc"].values for var in self.weather_vars],
            axis=1,
        )
        return torch.as_tensor(
            weather_msc,
            dtype=torch.float32,
        )

    def _build_doy_lookup(self):
        time = self.dataset.time_veg
        doy = time.dt.dayofyear.values

        is_leap = time.dt.is_leap_year.values
        doy[(doy == 60) & is_leap] = 59
        doy[(doy > 60) & is_leap] -= 1

        doy_lookup = (doy - 1) // self.sampling_days

        self.doy_lookup = torch.as_tensor(
            doy_lookup,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.training_pairs)

    def _validate_tensors(self, *tensors):
        for t in tensors:
            if torch.isnan(t).all() or torch.isinf(t).any():
                raise ValueError("Tensor contains only NaNs or infs")

    def _get_sample(self, sample_id, t0, include_metadata=False):
        veg = self.vegetation[sample_id]
        weather = self.weather[sample_id]
        hist_slice = slice(
            t0 - self.context_length,
            t0,
        )
        forecast_slice = slice(
            t0,
            t0 + self.prediction_length,
        )
        vegetation_history = veg[hist_slice]
        weather_history = weather[hist_slice]
        vegetation_forecast = veg[forecast_slice]
        weather_forecast = weather[forecast_slice]
        hist_doy = self.doy_lookup[hist_slice]
        forecast_doy = self.doy_lookup[forecast_slice]
        self._validate_tensors(
            vegetation_history,
            vegetation_forecast,
            weather_forecast,
        )
        sample = {
            "vegetation_history": vegetation_history,
            "weather_history": weather_history,
            "vegetation_forecast": vegetation_forecast,
            "weather_forecast": weather_forecast,
            "hist_doy": hist_doy,
            "forecast_doy": forecast_doy,
        }
        # if include_metadata:
        sample.update(
            {
                "sample_id": torch.tensor(sample_id),
                "location": self.locations[sample_id],
                "forecast_origin": torch.tensor(t0),
            }
        )
        return sample


class ForecastingTrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = Compose(
            [
                RandomFeatureMask(
                    keys=[
                        "weather_history",
                        "vegetation_history",
                    ],
                    p_feature=0.2,
                ),
                RandomGaussianJitter(
                    keys=[
                        "weather_history",
                        "vegetation_history",
                    ],
                    p_feature=0.5,
                    sigma=0.05,
                ),
            ]
        )

    def __getitem__(self, index):
        try:
            sample_id, t0 = self.training_pairs[index]
            sample = self._get_sample(sample_id, t0)

            if self.transform is not None:
                sample = self.transform(sample)
            return sample

        except Exception as e:
            return None


class ForecastingValDataset(BaseDataset):
    def __getitem__(self, index):
        try:
            sample_id, t0 = self.training_pairs[index]

            sample = self._get_sample(
                sample_id,
                t0,
                include_metadata=True,
            )

            return sample
        except Exception as e:
            return None
