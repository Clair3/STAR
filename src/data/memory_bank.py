import torch
from torch.utils.data import Dataset


class MemoryBank(Dataset):

    def __init__(self, base_dataset, retrieval_path):

        self.base_dataset = base_dataset

        raft = torch.load(
            retrieval_path,
        )

        self.retrieval = raft["retrieval"]
        self.weights = raft["weights"]
        self.pearson = raft.get("pearson")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        sample = self.base_dataset[index]

        if sample is None:
            return None

        retrieval = self.retrieval[index]
        weights = self.weights[index]

        # Backward-compatible names used by the current forecasting model.
        sample["retrieval"] = retrieval
        sample["retrieval_samples"] = retrieval
        sample["retrieval_weights"] = weights

        if self.pearson is not None:
            sample["retrieval_pearson"] = self.pearson[index]

        return sample
