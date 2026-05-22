# --- Data Configuration ---
# InstaNovo was trained with the top 200 peaks per spectrum; matching
# here keeps the frozen backbone in-distribution.
MAX_PEAKS = 200
MAX_PEPTIDE_LEN = 42
NUM_AA = 26
RADIANT_BASE = 10000.0
SEED = 42

# --- Model Architecture ---
D_MODEL = 128
N_HEADS = 4
D_FF = 768
N_LAYERS = 4
EMBED_DIM = 96
DROPOUT = 0.1

# --- Training Schedule ---
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
INIT_TEMP = 0.07
