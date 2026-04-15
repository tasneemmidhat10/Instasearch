import torch
from ..utils.constants import DEVICE

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