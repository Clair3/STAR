"""
Prepare a lightweight RAFT-style retrieval dataset for EVI forecasting.

Setup
-----
Forecasting context:
    1 year = 73 observations (5-day sampling)

Retrieval key:
    last 90 days = 18 observations

Retrieval value:
    following prediction period = 18 observations

Retrieval candidates (all from training years):
    - temporal: same location, all training windows
    - spatial: sampled OTHER locations, all their training windows
    - spatiotemporal: union of the two pools
    - top-k by EVI-history cosine similarity (no seasonal restriction)

The forecasting context and retrieval context are deliberately different.

The resulting file contains:

    retrieval:
        [N_query, topm, pred_len, n_vegetation_channels]

    training_pairs:
        [(sample_id, t0), ...]

This is intended as a cheap retrieval-augmented forecasting baseline.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .datasets import ForecastingValDataset



# ============================================================
# Configuration
# ============================================================

DEFAULT_RETRIEVAL_DAYS = 90
DEFAULT_TOPM = 20
DEFAULT_TEMPERATURE = 0.1


def days_to_steps(days, sampling_days):
    return max(1, round(days / sampling_days))


# ============================================================
# Retrieval
# ============================================================


class Retriever:

    def __init__(
        self,
        retrieval_length,
        prediction_length,
        topm=20,
        temperature=0.1,
        batch_size=32,
        device="cuda",
        num_other_locations=128,
        seed=42,
    ):

        self.retrieval_length = retrieval_length
        self.prediction_length = prediction_length

        self.topm = topm
        self.temperature = temperature

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.num_other_locations = num_other_locations
        self.seed = seed
        self._other_location_cache = {}  # same sampled locations in both modes

        self.train_keys = None
        self.train_values = None

        self.location_ids = None
        self.t0s = None

        self.n_train = None
        self.n_locations = None
        self.n_per_location = None
        self.train_keys_by_location = None
        self.train_values_by_location = None
        self.train_t0s_by_location = None

    # --------------------------------------------------------
    # Build retrieval corpus
    # --------------------------------------------------------

    def prepare_corpus(self, dataset):

        keys = []
        values = []
        locations = []
        t0s = []

        print("Preparing retrieval corpus...")

        for sample_id, t0 in tqdm(dataset.training_pairs):

            veg = dataset.vegetation[sample_id]

            # ------------------------------------------------
            # Retrieval key:
            #
            # last 90 days before t0
            # ------------------------------------------------

            key = veg[t0 - self.retrieval_length : t0]

            # ------------------------------------------------
            # Retrieved value:
            #
            # future EVI after candidate t0
            # ------------------------------------------------

            value = veg[t0 : t0 + self.prediction_length]

            keys.append(key)
            values.append(value)

            locations.append(sample_id)
            t0s.append(t0)

        self.train_keys = torch.stack(keys).float()

        self.train_values = torch.stack(values).float()

        self.location_ids = torch.tensor(
            locations,
            dtype=torch.long,
        )

        self.t0s = torch.tensor(
            t0s,
            dtype=torch.long,
        )

        self.n_train = len(dataset.training_pairs)

        # BaseDataset generates training_pairs grouped by sample_id:
        # [(0, t0_0), ..., (0, t0_K), (1, t0_0), ...].
        # Create inexpensive views so a full query batch can be retrieved
        # by indexing directly with its location IDs.
        self.n_locations = dataset.num_samples
        if self.n_train % self.n_locations != 0:
            raise ValueError("Unequal number of retrieval windows per location")
        self.n_per_location = self.n_train // self.n_locations
        if not 0 < self.num_other_locations < self.n_locations:
            raise ValueError("num_other_locations must be between 1 and n_locations - 1")

        expected_locations = torch.arange(
            self.n_locations, dtype=torch.long
        ).repeat_interleave(self.n_per_location)
        if not torch.equal(self.location_ids, expected_locations):
            raise ValueError(
                "Training pairs must be grouped by contiguous location IDs; "
                "otherwise direct location indexing is unsafe."
            )

        self.train_keys_by_location = self.train_keys.reshape(
            self.n_locations,
            self.n_per_location,
            self.retrieval_length,
            -1,
        )
        self.train_values_by_location = self.train_values.reshape(
            self.n_locations,
            self.n_per_location,
            self.prediction_length,
            -1,
        )
        self.train_t0s_by_location = self.t0s.reshape(
            self.n_locations, self.n_per_location
        )

        if self.topm > self.n_per_location:
            raise ValueError(
                f"topm={self.topm} exceeds candidates per location "
                f"({self.n_per_location})"
            )

        print(f"Candidates per location: {self.n_per_location}")
        print(f"Retrieval keys: " f"{self.train_keys.shape}")

        print(f"Retrieval values: " f"{self.train_values.shape}")

        print(f"Number of candidates: " f"{self.n_train}")

    def _other_locations(self, location):
        """Reproducible sample, excluding the query location (cached by location)."""
        if location not in self._other_location_cache:
            generator = torch.Generator().manual_seed(self.seed + location)
            # Sample from [0, n_locations-2], then skip `location`.
            ids = torch.randperm(self.n_locations - 1, generator=generator)
            ids = ids[:self.num_other_locations]
            ids = ids + (ids >= location).long()
            self._other_location_cache[location] = ids
        return self._other_location_cache[location]

    # --------------------------------------------------------
    # Retrieve one batch
    # --------------------------------------------------------

    @torch.no_grad()
    def retrieve_batch(
        self,
        query_keys,
        query_locations,
        query_t0s,
        exclude_temporal_overlap=True,
        mode="temporal",
    ):
        """Return [B, topm, pred_len, C] futures, weights, corpus indices.

        `query_t0s` are relative to the query split, so disable temporal
        overlap masking for val/test (their years are later than training).
        """
        if mode not in ("temporal", "spatial", "spatiotemporal"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if self.train_keys_by_location is None:
            raise RuntimeError("Call prepare_corpus() before retrieval")

        location_ids = query_locations.to(device="cpu", dtype=torch.long)
        if ((location_ids < 0) | (location_ids >= self.n_locations)).any():
            raise IndexError("Query contains an unknown location ID")

        q = query_keys.to(self.device)
        q_t0 = query_t0s.to(self.device)
        B, L, C = q.shape
        if L != self.retrieval_length or C != self.train_keys.shape[-1]:
            raise ValueError("Query has incorrect retrieval length or channel count")

        # Only this candidate-location selection differs between modes.
        if mode == "temporal":
            candidate_locations = location_ids[:, None]  # [B, 1]
        else:
            others = torch.stack([
                self._other_locations(int(loc)) for loc in location_ids.tolist()
            ])  # [B, num_other_locations]
            candidate_locations = (
                others if mode == "spatial"
                else torch.cat((location_ids[:, None], others), dim=1)
            )

        # [B, selected_locations * windows_per_location, L, C]
        candidate_keys = self.train_keys_by_location[candidate_locations]
        candidate_keys = candidate_keys.flatten(1, 2).to(self.device)
        K = candidate_keys.shape[1]

        # Original cosine similarity, now against the selected candidate pool.
        q_flat = q.reshape(B, -1)
        c_flat = candidate_keys.reshape(B, K, -1)
        q_normalized = F.normalize(q_flat, dim=-1)
        c_normalized = F.normalize(c_flat, dim=-1)
        similarity = torch.bmm(
            c_normalized, q_normalized.unsqueeze(-1)
        ).squeeze(-1)

        if exclude_temporal_overlap:
            # Correct for training queries, where t0 shares the same time axis.
            candidate_t0s = self.train_t0s_by_location[candidate_locations]
            candidate_t0s = candidate_t0s.flatten(1, 2).to(self.device)
            candidate_end = candidate_t0s + self.prediction_length
            query_start = q_t0[:, None]
            query_end = query_start + self.prediction_length
            overlap = (candidate_t0s < query_end) & (candidate_end > query_start)
            similarity = similarity.masked_fill(overlap, float("-inf"))

        available = torch.isfinite(similarity).sum(dim=1)
        if (available < self.topm).any():
            bad_idx = (available < self.topm).nonzero(as_tuple=False)[0, 0]
            raise RuntimeError(
                f"Location {location_ids[bad_idx].item()}, "
                f"query t0={query_t0s[bad_idx].item()}: "
                f"only {available[bad_idx].item()} valid candidates "
                f"(topm={self.topm})."
            )

        top_values, top_indices = torch.topk(similarity, self.topm, dim=1)
        weights = F.softmax(top_values / self.temperature, dim=1)

        # Recover flat training-corpus indices; gather ONLY top-m futures.
        top_indices = top_indices.cpu()
        selected_locations = candidate_locations.gather(
            1, top_indices // self.n_per_location
        )
        candidate_indices = (
            selected_locations * self.n_per_location
            + top_indices % self.n_per_location
        )
        retrieved_values = self.train_values[candidate_indices]
        return retrieved_values.cpu(), weights.cpu(), candidate_indices

    # --------------------------------------------------------
    # Retrieve entire dataset
    # --------------------------------------------------------

    @torch.no_grad()
    def retrieve_dataset(
        self,
        dataset,
        exclude_temporal_overlap=True,
        mode="temporal",
        query_indices=None,
    ):
        # Optionally evaluate a small, identical subset across all modes.
        pairs = (
            dataset.training_pairs if query_indices is None
            else [dataset.training_pairs[i] for i in query_indices]
        )
        N = len(pairs)
        all_retrievals = []
        all_weights = []
        all_indices = []
        print(f"Computing {mode} retrievals for {N} samples...")

        for start in tqdm(range(0, N, self.batch_size)):
            batch_pairs = pairs[start:start + self.batch_size]
            keys = []
            locations = []
            t0s = []
            for sample_id, t0 in batch_pairs:
                veg = dataset.vegetation[sample_id]
                keys.append(veg[t0 - self.retrieval_length:t0])
                locations.append(sample_id)
                t0s.append(t0)

            retrieval, weights, indices = self.retrieve_batch(
                torch.stack(keys).float(),
                torch.tensor(locations, dtype=torch.long),
                torch.tensor(t0s, dtype=torch.long),
                exclude_temporal_overlap=exclude_temporal_overlap,
                mode=mode,
            )
            all_retrievals.append(retrieval)
            all_weights.append(weights)
            all_indices.append(indices)

        return (
            torch.cat(all_retrievals, dim=0),
            torch.cat(all_weights, dim=0),
            torch.cat(all_indices, dim=0),
        )


# ============================================================
# Dataset
# ============================================================


def build_dataset(
    data_config,
    years,
    context_days,
    prediction_days,
):

    return ForecastingValDataset(
        dataset_path=data_config["path"],
        vegetation_indices=data_config["vegetation"]["variables"],
        weather_vars=data_config["weather"]["variables"],
        years=years,
        context_days=context_days,
        prediction_days=prediction_days,
        sampling_days=data_config["sampling_days"],
    )


def save_retrieval(
    output_file,
    retrieval,
    weights,
    dataset,
    retrieval_length,
    prediction_length,
    args,
    data_config,
    mode,
    candidate_indices,
    query_indices,
):

    torch.save(
        {
            "retrieval": retrieval,
            "weights": weights,
            "training_pairs": [dataset.training_pairs[i] for i in query_indices],
            "query_indices": query_indices,
            "candidate_indices": candidate_indices,
            "mode": mode,
            "context_length": dataset.context_length,
            "retrieval_length": retrieval_length,
            "prediction_length": prediction_length,
            "topm": args.topm,
            "temperature": args.temperature,
            "vegetation_variables": (data_config["vegetation"]["variables"]),
            "retrieval_type": f"EVI history cosine, {mode}, training corpus only",
        },
        output_file,
    )

    print(f"Saved {output_file}")


# ============================================================
# Main
# ============================================================


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config",
        type=str,
        default="/Net/Groups/BGI/scratch/crobin/PythonProjects/SpatioTemporalRAG/configs/data_config_VIIRS_passage.yaml",
        # required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Net/Groups/BGI/scratch/crobin/PythonProjects/SpatioTemporalRAG/RetrievalDatasets/",
    )

    parser.add_argument(
        "--context_days",
        type=int,
        default=365,
    )

    parser.add_argument(
        "--prediction_days",
        type=int,
        default=92,
    )

    parser.add_argument(
        "--retrieval_days",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--topm",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--modes", nargs="+",
        choices=["temporal", "spatial", "spatiotemporal"],
        default=["temporal", "spatial", "spatiotemporal"],
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"],
        default=["train", "test"],
    )
    parser.add_argument("--num_other_locations", type=int, default=128)
    parser.add_argument("--max_queries", type=int, default=None) #default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    # --------------------------------------------------------
    # Load YAML
    # --------------------------------------------------------

    import yaml

    with open(
        args.data_config,
        "r",
    ) as f:

        data_config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------
    print("\nLoading training dataset...")

    train_dataset = build_dataset(
        data_config,
        years=data_config["forecasting"]["train"],
        context_days=args.context_days,
        prediction_days=args.prediction_days,
    )

    retrieval_length = days_to_steps(
        args.retrieval_days,
        data_config["sampling_days"],
    )

    prediction_length = train_dataset.prediction_length

    channels = len(data_config["vegetation"]["variables"])

    print("\nRetrieval configuration:")

    print(f"  forecasting context : " f"{train_dataset.context_length} steps")

    print(f"  retrieval context   : " f"{retrieval_length} steps")

    print(f"  prediction length   : " f"{prediction_length} steps")

    print(f"  channels            : " f"{channels}")

    print(f"  top-m               : " f"{args.topm}")

    print(f"  temperature         : " f"{args.temperature}")

    print("\nImportant:")

    print("Retrieval modes:", ", ".join(args.modes))
    print("Spatial candidates: all training periods, no season filter")

    print("Training retrieval excludes overlapping forecast periods.")
    print("Validation/test retrieve from earlier training years without index-based masking.")

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------
    retriever = Retriever(
        retrieval_length=retrieval_length,
        prediction_length=train_dataset.prediction_length,
        topm=args.topm,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=args.device,
        num_other_locations=args.num_other_locations,
        seed=args.seed,
    )

    retriever.prepare_corpus(train_dataset)

    # --------------------------------------------------------
    # Compute retrievals; reuse the same validation queries for every mode.
    # --------------------------------------------------------
    for split in args.splits:
        if split == "train":
            dataset = train_dataset
        else:
            config_split = "validation" if split == "val" else "test"
            print(f"\nLoading {split} dataset...")
            dataset = build_dataset(
                data_config,
                years=data_config["forecasting"][config_split],
                context_days=args.context_days,
                prediction_days=args.prediction_days,
            )
            # Check chronological separation using real xarray time coordinates.
            first_query_t0 = dataset.training_pairs[0][1]
            if not (
                train_dataset.dataset.time_veg.values[-1]
                < dataset.dataset.time_veg.values[first_query_t0]
            ):
                raise ValueError("Training data overlap the query forecast dates")

        # Random subset across locations/windows; fixed for all three modes.
        if args.max_queries is None or args.max_queries <= 0 or args.max_queries >= len(dataset.training_pairs):
            query_indices = range(len(dataset.training_pairs))
        else:
            generator = torch.Generator().manual_seed(args.seed)
            query_indices = torch.randperm(
                len(dataset.training_pairs), generator=generator
            )[:args.max_queries].sort().values.tolist()

        for mode in args.modes:
            print(f"\nRetrieving {split} | {mode} ...")
            retrieval, weights, candidate_indices = retriever.retrieve_dataset(
                dataset,
                exclude_temporal_overlap=(split == "train"),
                mode=mode,
                query_indices=query_indices,
            )
            mode_dir = output_dir / "POC" / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            save_retrieval(
                mode_dir / f"{split}.pt",
                retrieval, weights, dataset,
                retrieval_length, prediction_length, args, data_config,
                mode, candidate_indices, query_indices,
            )

    print("\n========================================")

    print("RAFT retrieval preprocessing complete.")


if __name__ == "__main__":
    main()
