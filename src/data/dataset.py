import torch
from torch.utils.data import Dataset

class SpectraPeptideDataset(Dataset):
    def __init__(self, specs, peps, pres):
        self.specs = torch.tensor(specs, dtype=torch.float32)
        self.peps = torch.tensor(peps, dtype=torch.int64)
        self.pres = torch.tensor(pres, dtype=torch.float32)
    def __len__(self): return len(self.specs)
    def __getitem__(self, i): return self.specs[i], self.peps[i], self.pres[i]