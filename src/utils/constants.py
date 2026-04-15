import torch

# --- Global Configuration Constants ---
MAX_PEAKS = 500
MAX_PEPTIDE_LEN = 42
ENCODING_DIM = 64
RADIANT_BASE = 10000.0
SEED = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")