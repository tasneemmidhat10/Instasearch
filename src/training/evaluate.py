import torch
from ..utils.constants import DEVICE

@torch.no_grad()
def check_feature_collapse(model_spec, model_pep, dataloader, device):
    model_spec.eval()
    model_pep.eval()

    with torch.no_grad():
        # 1. Grab a single batch
        specs, peps, pres = next(iter(dataloader))
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)

        # 2. Generate embeddings
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                z_s = model_spec(specs, pres)
                z_p = model_pep(peps)
        else:
            z_s = model_spec(specs, pres)
            z_p = model_pep(peps)

        # 3. Standard Deviation of the embedding matrix
        # We calculate the standard deviation along the batch dimension
        # for each feature, then take the mean across all features.
        std_s = z_s.std(dim=0).mean().item()
        std_p = z_p.std(dim=0).mean().item()

        # 4. Mean Cosine Similarity (Off-diagonal)
        # Because outputs are L2-normalized, dot product == cosine similarity
        sim_matrix_s = torch.mm(z_s, z_s.t())
        sim_matrix_p = torch.mm(z_p, z_p.t())

        # Create a mask to ignore the diagonal (self-similarity is always 1.0)
        batch_size = z_s.size(0)
        off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        mean_sim_s = sim_matrix_s[off_diag_mask].mean().item()
        mean_sim_p = sim_matrix_p[off_diag_mask].mean().item()

        # 5. Print Diagnostics
        print("--- Feature Collapse Diagnostics ---")
        print(f"Spectrum Embeddings - Mean Std Dev: {std_s:.4f} (Ideal: > 0.0)")
        print(f"Peptide Embeddings  - Mean Std Dev: {std_p:.4f} (Ideal: > 0.0)")
        print(f"Spectrum vs Spectrum - Mean Off-Diag Cosine Sim: {mean_sim_s:.4f} (Ideal: ~ 0.0)")
        print(f"Peptide vs Peptide   - Mean Off-Diag Cosine Sim: {mean_sim_p:.4f} (Ideal: ~ 0.0)")

        # Simple heuristic warning
        if mean_sim_s > 0.9 or mean_sim_p > 0.9:
            print("\n⚠️ WARNING: High similarity between random items detected. The model may be suffering from feature collapse.")
        else:
            print("\n✅ HEALTHY: Embeddings show good variance and low average similarity between random items.")

@torch.no_grad()
def evaluate_top_k_retrieval(model_spec, model_pep, dataloader, device, k_vals=[1, 5, 10]):
    """
    Evaluates Top-K retrieval accuracy for spectrum-to-peptide matching.
    """
    model_spec.eval()
    model_pep.eval()

    all_z_s = []
    all_z_p = []

    print("Extracting embeddings for retrieval evaluation...")
    for specs, peps, pres in dataloader:
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)

        # Generate and store embeddings
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                z_s = model_spec(specs, pres)
                z_p = model_pep(peps)
        else:
            z_s = model_spec(specs, pres)
            z_p = model_pep(peps)

        all_z_s.append(z_s)
        all_z_p.append(z_p)

    # Concatenate all batches into single large tensors
    all_z_s = torch.cat(all_z_s, dim=0)
    all_z_p = torch.cat(all_z_p, dim=0)

    num_samples = all_z_s.size(0)
    print(f"Calculating Top-K accuracy across {num_samples} samples...")

    # Calculate full NxN similarity matrix
    # sim_matrix[i, j] = cosine similarity between spectrum i and peptide j
    sim_matrix = torch.mm(all_z_s, all_z_p.t())

    # Get the indices of the top K most similar peptides for each spectrum
    max_k = max(k_vals)
    _, top_k_indices = sim_matrix.topk(max_k, dim=1, largest=True, sorted=True)

    # The correct peptide for spectrum i is at index i.
    # We want to check if 'i' is in the top K indices for spectrum 'i'
    correct_indices = torch.arange(num_samples, device=device).unsqueeze(1)

    results = {}
    for k in k_vals:
        # Check if the correct index is within the first k predictions
        matches = (top_k_indices[:, :k] == correct_indices).any(dim=1)
        accuracy = matches.float().mean().item()
        results[f'Top-{k}'] = accuracy
        print(f"Top-{k:2d} Accuracy: {accuracy * 100:.2f}%")

    return results