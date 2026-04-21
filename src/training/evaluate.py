import torch
from ..utils.constants import DEVICE

@torch.no_grad()
def check_feature_collapse(model_spec, model_pep, dataloader, device, max_batches=10):
    model_spec.eval()
    model_pep.eval()

    all_z_s = []
    all_z_p = []

    for i, (specs, peps, pres) in enumerate(dataloader):
        if i >= max_batches:
            break
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                z_s = model_spec(specs, pres)
                z_p = model_pep(peps)
        else:
            z_s = model_spec(specs, pres)
            z_p = model_pep(peps)

        all_z_s.append(z_s.cpu())
        all_z_p.append(z_p.cpu())

    all_z_s = torch.cat(all_z_s, dim=0)
    all_z_p = torch.cat(all_z_p, dim=0)

    std_s = all_z_s.std(dim=0).mean().item()
    std_p = all_z_p.std(dim=0).mean().item()

    # Cap samples for cosine similarity to avoid O(n^2) memory blowup
    cap = min(512, all_z_s.size(0))
    z_s_cap = all_z_s[:cap]
    z_p_cap = all_z_p[:cap]

    sim_matrix_s = torch.mm(z_s_cap, z_s_cap.t())
    sim_matrix_p = torch.mm(z_p_cap, z_p_cap.t())

    off_diag_mask = ~torch.eye(cap, dtype=torch.bool)
    mean_sim_s = sim_matrix_s[off_diag_mask].mean().item()
    mean_sim_p = sim_matrix_p[off_diag_mask].mean().item()

    print("--- Feature Collapse Diagnostics ---")
    print(f"Spectrum Embeddings - Mean Std Dev: {std_s:.4f} (Ideal: > 0.0)")
    print(f"Peptide Embeddings  - Mean Std Dev: {std_p:.4f} (Ideal: > 0.0)")
    print(f"Spectrum vs Spectrum - Mean Off-Diag Cosine Sim: {mean_sim_s:.4f} (Ideal: ~ 0.0)")
    print(f"Peptide vs Peptide   - Mean Off-Diag Cosine Sim: {mean_sim_p:.4f} (Ideal: ~ 0.0)")

    if mean_sim_s > 0.9 or mean_sim_p > 0.9:
        print("\nWARNING: High similarity between random items detected. The model may be suffering from feature collapse.")
    else:
        print("\nHEALTHY: Embeddings show good variance and low average similarity between random items.")

@torch.no_grad()
def evaluate_top_k_retrieval(model_spec, model_pep, dataloader, device, k_vals=[1, 5, 10], chunk_size=512):
    """
    Evaluates Top-K retrieval accuracy for spectrum-to-peptide matching.
    Chunked similarity computation avoids materializing the full N×N matrix.
    """
    model_spec.eval()
    model_pep.eval()

    all_z_s = []
    all_z_p = []

    print("Extracting embeddings for retrieval evaluation...")
    for specs, peps, pres in dataloader:
        specs, peps, pres = specs.to(device), peps.to(device), pres.to(device)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                z_s = model_spec(specs, pres)
                z_p = model_pep(peps)
        else:
            z_s = model_spec(specs, pres)
            z_p = model_pep(peps)

        # Move to CPU immediately to free GPU memory
        all_z_s.append(z_s.cpu())
        all_z_p.append(z_p.cpu())

    all_z_s = torch.cat(all_z_s, dim=0)  # [N, D]
    all_z_p = torch.cat(all_z_p, dim=0)  # [N, D]

    num_samples = all_z_s.size(0)
    print(f"Calculating Top-K accuracy across {num_samples} samples (chunk_size={chunk_size})...")

    max_k = max(k_vals)
    correct_count = {k: 0 for k in k_vals}

    # Process spectrum embeddings in chunks; compute [chunk, N] similarity at a time
    # instead of the full [N, N] matrix
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        z_s_chunk = all_z_s[start:end]

        sim_chunk = torch.mm(z_s_chunk, all_z_p.t())  # [chunk, N]
        _, top_k_indices = sim_chunk.topk(max_k, dim=1, largest=True, sorted=True)

        correct_indices = torch.arange(start, end).unsqueeze(1)

        for k in k_vals:
            matches = (top_k_indices[:, :k] == correct_indices).any(dim=1)
            correct_count[k] += matches.sum().item()

    results = {}
    for k in k_vals:
        accuracy = correct_count[k] / num_samples
        results[f'Top-{k}'] = accuracy
        print(f"Top-{k:2d} Accuracy: {accuracy * 100:.2f}%")

    return results
