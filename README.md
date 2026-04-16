<<<<<<< HEAD
# InstaSearch Source Code (`src/`)

## Overview

This directory contains the core source code for **InstaSearch**, a machine learning-based proteomics search engine. The project implements a contrastive learning approach inspired by CLIP (Contrastive Language-Image Pretraining) to learn joint embeddings for mass spectrometry spectra and peptide sequences. This enables efficient and accurate peptide-spectrum matching for database search in proteomics.

The system consists of dual encoders: a **spectrum encoder** that processes mass spectrometry data and a **peptide encoder** that handles amino acid sequences. These encoders are trained jointly using a contrastive loss to maximize similarity between matching spectrum-peptide pairs while minimizing similarity for non-matching pairs.

Key features:
- Transformer-based architectures for both encoders
- Contrastive learning with temperature-scaled softmax loss
- Support for large-scale proteomics datasets
- Modular design for easy extension and experimentation

## Project Structure

```
src/
├── __init__.py                 # Package initialization
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset for spectra-peptide pairs
│   └── preprocess.py           # Data preprocessing utilities
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── joint_model.py          # Shared transformer blocks and projection heads
│   ├── peptide_encoder.py      # Peptide sequence encoder
│   └── spectrum_encoder.py     # Mass spectrometry spectrum encoder
├── retrieval/                  # Search and retrieval components
│   ├── index.py                # Building and managing search indices
│   ├── rerank.py               # Re-ranking search results
│   └── search.py               # Core search functionality
├── retreieval/                 # (Note: Likely a typo - duplicate of retrieval/)
│   ├── __init__.py
│   └── search.py
├── training/                   # Training and evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py             # Model evaluation utilities
│   ├── loss.py                 # Contrastive loss functions
│   └── train.py                # Training loop implementation
└── utils/                      # Utility functions and configurations
    ├── __init__.py
    ├── config.py               # Configuration parameters
    ├── constants.py            # Constants and hyperparameters
    └── visualization.py        # Plotting and visualization tools
```

## Installation and Requirements

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for training)

### Dependencies
Install required packages using pip:
```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib seaborn
pip install transformers huggingface-hub
```

For development:
```bash
pip install pytest black isort mypy
```

### Setup
1. Clone the repository and navigate to the project directory
2. Install dependencies as above
3. Configure paths in `utils/config.py` for your data directories

## Usage

### Training a Model

```python
from src.training.train import train_epoch
from src.models.peptide_encoder import PeptideEncoder
from src.models.spectrum_encoder import SpectrumEncoder
from src.training.loss import CLIPContrastiveLoss
from src.data.dataset import SpectraPeptideDataset
import torch

# Initialize models
peptide_encoder = PeptideEncoder()
spectrum_encoder = SpectrumEncoder()

# Load data
dataset = SpectraPeptideDataset(spectra, peptides, precursors)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Setup training
loss_fn = CLIPContrastiveLoss()
optimizer = torch.optim.Adam([
    {'params': peptide_encoder.parameters()},
    {'params': spectrum_encoder.parameters()}
], lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        spectrum_encoder, peptide_encoder, 
        dataloader, loss_fn, optimizer, scaler=None
    )
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
```

### Performing Search

```python
from src.retrieval.search import search_spectra
from src.models.peptide_encoder import PeptideEncoder
from src.models.spectrum_encoder import SpectrumEncoder

# Load trained models
peptide_encoder = PeptideEncoder()
spectrum_encoder = SpectrumEncoder()
peptide_encoder.load_state_dict(torch.load('peptide_encoder.pth'))
spectrum_encoder.load_state_dict(torch.load('spectrum_encoder.pth'))

# Load peptide database
peptide_db = load_peptide_database()  # Your implementation

# Search
query_spectra = load_query_spectra()  # Your spectra
results = search_spectra(query_spectra, peptide_db, 
                        spectrum_encoder, peptide_encoder)
```

## Module Descriptions

### Data Module (`data/`)

- **`dataset.py`**: Defines `SpectraPeptideDataset`, a PyTorch Dataset class for loading and batching spectrum-peptide pairs with precursor information.
- **`preprocess.py`**: Contains utilities for preprocessing mass spectrometry data, including peak filtering, normalization, and peptide tokenization.

### Models Module (`models/`)

- **`joint_model.py`**: Shared components used by both encoders, including `TransformerEncoderBlock` for attention layers and `ProjectionHead` for final embedding projection.
- **`peptide_encoder.py`**: Implements `PeptideEncoder`, a transformer-based model that encodes amino acid sequences into fixed-dimensional embeddings using positional encoding and a CLS token.
- **`spectrum_encoder.py`**: Implements `SpectrumEncoder`, processing mass spectrometry peak data through convolutional and transformer layers.

### Retrieval Module (`retrieval/`)

- **`index.py`**: Functions for building and managing searchable indices of peptide embeddings for efficient nearest-neighbor search.
- **`rerank.py`**: Implements re-ranking algorithms to improve search results using additional scoring functions.
- **`search.py`**: Core search functionality, including similarity computation and candidate ranking.

### Training Module (`training/`)

- **`evaluate.py`**: Evaluation metrics and validation functions for assessing model performance on held-out data.
- **`loss.py`**: Defines `CLIPContrastiveLoss`, the contrastive loss function used for joint training of spectrum and peptide encoders.
- **`train.py`**: Contains the main training loop with support for mixed precision training and gradient scaling.

### Utils Module (`utils/`)

- **`config.py`**: Configuration file containing hyperparameters, model dimensions, and training settings.
- **`constants.py`**: Defines constants such as device settings, amino acid vocabularies, and file paths.
- **`visualization.py`**: Utilities for plotting training curves, embedding visualizations, and search result analysis.

## Configuration

Key parameters are defined in `utils/config.py`:

- Model architecture: `D_MODEL`, `N_HEADS`, `D_FF`, `N_LAYERS`
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- Data: `MAX_PEPTIDE_LEN`, `EMBED_DIM`
- Loss: `INIT_TEMP` (initial temperature for contrastive loss)

Modify these values to experiment with different model configurations.

## Contributing

1. Follow the existing code style (use `black` for formatting)
2. Add type hints where possible
3. Write tests for new functionality in the `tests/` directory
4. Update this README when adding new modules or features
5. Ensure compatibility with both CPU and GPU training

## License

This project is part of the InstaSearch proteomics toolkit. See the main repository for licensing information.

## References

- CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- InstaNovo: Deep learning-enabled de novo peptide sequencing (Tran et al., 2023)
- Contrastive learning for peptide-spectrum matching in proteomics</content>
<parameter name="filePath">c:\Users\tasne\Desktop\InstaSearch Project\project\src\README.md
=======
# InstaSearch Source Code (`src/`)

## Overview

This directory contains the core source code for **InstaSearch**, a machine learning-based proteomics search engine. The project implements a contrastive learning approach inspired by CLIP (Contrastive Language-Image Pretraining) to learn joint embeddings for mass spectrometry spectra and peptide sequences. This enables efficient and accurate peptide-spectrum matching for database search in proteomics.

The system consists of dual encoders: a **spectrum encoder** that processes mass spectrometry data and a **peptide encoder** that handles amino acid sequences. These encoders are trained jointly using a contrastive loss to maximize similarity between matching spectrum-peptide pairs while minimizing similarity for non-matching pairs.

Key features:
- Transformer-based architectures for both encoders
- Contrastive learning with temperature-scaled softmax loss
- Support for large-scale proteomics datasets
- Modular design for easy extension and experimentation

## Project Structure

```
src/
├── __init__.py                 # Package initialization
├── data/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset for spectra-peptide pairs
│   └── preprocess.py           # Data preprocessing utilities
├── models/                     # Neural network models
│   ├── __init__.py
│   ├── joint_model.py          # Shared transformer blocks and projection heads
│   ├── peptide_encoder.py      # Peptide sequence encoder
│   └── spectrum_encoder.py     # Mass spectrometry spectrum encoder
├── retrieval/                  # Search and retrieval components
│   ├── index.py                # Building and managing search indices
│   ├── rerank.py               # Re-ranking search results
│   └── search.py               # Core search functionality
├── training/                   # Training and evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py             # Model evaluation utilities
│   ├── loss.py                 # Contrastive loss functions
│   └── train.py                # Training loop implementation
└── utils/                      # Utility functions and configurations
    ├── __init__.py
    ├── config.py               # Configuration parameters
    ├── constants.py            # Constants and hyperparameters
    └── visualization.py        # Plotting and visualization tools
```

## Installation and Requirements

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for training)

### Dependencies
Install required packages using pip:
```bash
pip install torch 
pip install numpy pandas scikit-learn matplotlib seaborn
pip install transformers huggingface-hub
```


### Setup
1. Clone the repository and navigate to the project directory
2. Install dependencies as above
3. Configure paths in `utils/config.py` for your data directories

## Usage

### Training a Model

```python
from src.training.train import train_epoch
from src.models.peptide_encoder import PeptideEncoder
from src.models.spectrum_encoder import SpectrumEncoder
from src.training.loss import CLIPContrastiveLoss
from src.data.dataset import SpectraPeptideDataset
import torch

# Initialize models
peptide_encoder = PeptideEncoder()
spectrum_encoder = SpectrumEncoder()

# Load data
dataset = SpectraPeptideDataset(spectra, peptides, precursors)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Setup training
loss_fn = CLIPContrastiveLoss()
optimizer = torch.optim.Adam([
    {'params': peptide_encoder.parameters()},
    {'params': spectrum_encoder.parameters()}
], lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        spectrum_encoder, peptide_encoder, 
        dataloader, loss_fn, optimizer, scaler=None
    )
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
```


## Module Descriptions

### Data Module (`data/`)

- **`dataset.py`**: Defines `SpectraPeptideDataset`, a PyTorch Dataset class for loading and batching spectrum-peptide pairs with precursor information.
- **`preprocess.py`**: Contains utilities for preprocessing mass spectrometry data, including peak filtering, normalization, and peptide tokenization.

### Models Module (`models/`)

- **`joint_model.py`**: Shared components used by both encoders, including `TransformerEncoderBlock` for attention layers and `ProjectionHead` for final embedding projection.
- **`peptide_encoder.py`**: Implements `PeptideEncoder`, a transformer-based model that encodes amino acid sequences into fixed-dimensional embeddings using positional encoding and a CLS token.
- **`spectrum_encoder.py`**: Implements `SpectrumEncoder`, processing mass spectrometry peak data through convolutional and transformer layers.

### Retrieval Module (`retrieval/`)

- **`index.py`**: Functions for building and managing searchable indices of peptide embeddings for efficient nearest-neighbor search.
- **`rerank.py`**: Implements re-ranking algorithms to improve search results using additional scoring functions.
- **`search.py`**: Core search functionality, including similarity computation and candidate ranking.

### Training Module (`training/`)

- **`evaluate.py`**: Evaluation metrics and validation functions for assessing model performance on held-out data.
- **`loss.py`**: Defines `CLIPContrastiveLoss`, the contrastive loss function used for joint training of spectrum and peptide encoders.
- **`train.py`**: Contains the main training loop with support for mixed precision training and gradient scaling.

### Utils Module (`utils/`)

- **`config.py`**: Configuration file containing hyperparameters, model dimensions, and training settings.
- **`constants.py`**: Defines constants such as device settings, amino acid vocabularies, and file paths.
- **`visualization.py`**: Utilities for plotting training curves, embedding visualizations, and search result analysis.

## Configuration

Key parameters are defined in `utils/config.py`:

- Model architecture: `D_MODEL`, `N_HEADS`, `D_FF`, `N_LAYERS`
- Training: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`
- Data: `MAX_PEPTIDE_LEN`, `EMBED_DIM`
- Loss: `INIT_TEMP` (initial temperature for contrastive loss)

Modify these values to experiment with different model configurations.

## Contributing

1. Follow the existing code style (use `black` for formatting)
2. Add type hints where possible
3. Write tests for new functionality in the `tests/` directory
4. Update this README when adding new modules or features
5. Ensure compatibility with both CPU and GPU training

## License

This project is part of the InstaSearch proteomics toolkit. See the main repository for licensing information.

## References

- CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- InstaNovo: Deep learning-enabled de novo peptide sequencing (Elof et al., 2023)
>>>>>>> 01bba729f1a19aaa8c5e3a83f1a030497e8d7381
