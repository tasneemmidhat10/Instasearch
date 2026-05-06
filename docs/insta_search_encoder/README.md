# InstaSearchSpectrumEncoder

Documentation for the spectrum tower of the dual-encoder contrastive retrieval pipeline.

`InstaSearchSpectrumEncoder` wraps InstaNovo's pretrained transformer encoder (weights frozen) and attaches a trainable MLP projection head that maps its `d_model` representation into the shared contrastive embedding space. It is one half of a CLIP-style dual encoder; the other half is the trainable `PeptideEncoder`.

## Contents

- [architecture.md](architecture.md) — what the module is, how data flows through it, why the backbone is frozen.
- [api.md](api.md) — method-by-method reference for every function in the class.
- [training_integration.md](training_integration.md) — how the encoder is constructed, how it plugs into the optimiser and training loop.
- [preprocessing.md](preprocessing.md) — the input contract the frozen backbone expects, and why our preprocessing mirrors InstaNovo's training distribution.

## Where it lives

- Class definition: notebook cell id `7265d906` in [training_workflow_dual_encoders.ipynb](../../training_workflow_dual_encoders.ipynb).
- Instantiation and optimiser wiring: cell id `67ee7bd2`.
- Training loop that consumes it: cell id `7135fe91`.
