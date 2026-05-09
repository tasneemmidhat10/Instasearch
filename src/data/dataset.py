import torch
from torch.utils.data import Dataset


class SpectraPeptideDataset(Dataset):
    """Dual-encoder training dataset.

    If `decoy_peps` is provided, `__getitem__` returns 4-tuples
    (spec, pep, pre, decoy); otherwise 3-tuples (spec, pep, pre).
    The training loop dispatches on the tuple length.
    """

    def __init__(self, specs, peps, pres, decoy_peps=None):
        self.specs = torch.as_tensor(specs, dtype=torch.float32)
        self.peps  = torch.as_tensor(peps,  dtype=torch.int64)
        self.pres  = torch.as_tensor(pres,  dtype=torch.float32)
        self.decoy_peps = (torch.as_tensor(decoy_peps, dtype=torch.int64)
                           if decoy_peps is not None else None)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        if self.decoy_peps is None:
            return self.specs[i], self.peps[i], self.pres[i]
        return self.specs[i], self.peps[i], self.pres[i], self.decoy_peps[i]
