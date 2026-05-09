"""Helpers for loading the vendored InstaNovo backbone without modifying it."""
import os
import sys
import torch

# Ensure the vendored InstaNovo package is importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_INSTANOVO_PATH = os.path.join(_PROJECT_ROOT, "InstaNovo")
if _INSTANOVO_PATH not in sys.path:
    sys.path.insert(0, _INSTANOVO_PATH)

from instanovo.transformer.model import InstaNovo  # noqa: E402


def load_instanovo_backbone(checkpoint: str, device: torch.device,
                            freeze: bool = True):
    """Load an InstaNovo checkpoint and return (model, config, d_model).

    `checkpoint` may be either a local .ckpt path or a model id understood by
    `InstaNovo.from_pretrained` (e.g. "instanovo-v1.2.0").
    """
    model, config = InstaNovo.from_pretrained(checkpoint)
    model = model.to(device)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    d_model = int(config["dim_model"])
    return model, config, d_model
