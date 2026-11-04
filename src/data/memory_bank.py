import torch
from torch.utils.data import Dataset


class MemoryBank(Dataset):
    
    def __init__(self, base_dataset, retrieval_path):
        self.base_dataset = base_dataset
        bank = torch.load(
            retrieval_path,
            map_location="cpu",
            weights_only=False,
        )
        self.retrieval =bank["retrieval"]
        self.weights = bank["weights"]
        if len(self.retrieval) != len(self.base_dataset):
            raise ValueError("Retrieval and dataset lengths do not match.")
        if bank["training_pairs"] != self.base_dataset.training_pairs:
            raise ValueError("Retrieval and dataset samples are not aligned.")
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, index):
        sample = self.base_dataset[index]
        if sample is None:
            return None
        sample["retrieval"] = self.retrieval[index]
        sample["retrieval_weights"] = self.weights[index]
        return sample