import torch
from torch.utils.data import Dataset

from core.preprocess.instances import ListInstance, ValueInstance


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class CustomCollate:
    @staticmethod
    def collate(instances):
        batch = {}
        for instance in instances:
            for k, v in instance.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        for k, instances in batch.items():
            if isinstance(instances[0], ListInstance):
                max_len = max([len(instance) for instance in instances])
                batch[k] = torch.stack(
                    [instance.to_padded_tensor(max_len) for instance in instances],
                    dim=0
                )
            elif isinstance(instances[0], ValueInstance):
                batch[k] = torch.stack([
                    instance.to_tensor() for instance in instances
                ], dim=0)
        return batch
