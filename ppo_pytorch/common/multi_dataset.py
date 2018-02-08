import numpy as np
import torch.utils.data as data


class MultiDataset(data.Dataset):
    """
    Dataset without labels
    """
    def __init__(self, *colls):
        self.colls = colls

    def __getitem__(self, index):
        return [self.process_point(coll[index]) for coll in self.colls]

    @staticmethod
    def process_point(x):
        return np.array([x], dtype=np.float32) if isinstance(x, float) else x

    def __len__(self):
        return min([len(c) for c in self.colls])