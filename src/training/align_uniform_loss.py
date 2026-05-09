import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignUniformLoss(nn.Module):
    """Wang & Isola alignment + uniformity loss with VICReg variance term and
    optional decoy-margin term. Lifted from
    `dual_encoders_alignmentand_UniformityLoss_withPTMs.ipynb`.

    forward returns (loss, acc, diagnostics) — matching the unified contract
    used by the dual-encoder training loop.
    """

    def __init__(self, alpha: float = 2.0, t: float = 2.0,
                 lam: float = 1.0, lam_var: float = 1.0,
                 decoy_weight: float = 0.3, decoy_margin: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.t = t
        self.lam = lam
        self.lam_var = lam_var
        self.decoy_weight = decoy_weight
        self.decoy_margin = decoy_margin

    def variance_loss(self, z):
        gamma = 1.0 / math.sqrt(z.size(1))
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return F.relu(gamma - std).mean()

    def forward(self, z_spec, z_pep, z_decoy=None):
        B = z_spec.size(0)

        L_align = (z_spec - z_pep).norm(dim=1).pow(self.alpha).mean()

        L_unif_s = torch.pdist(z_spec, p=2).pow(2).mul(-self.t).exp().mean().log()
        L_unif_p = torch.pdist(z_pep,  p=2).pow(2).mul(-self.t).exp().mean().log()
        L_uniform = (L_unif_s + L_unif_p) / 2

        L_decoy = torch.tensor(0.0, device=z_spec.device)
        if z_decoy is not None:
            pos_dist = (z_spec - z_pep).norm(dim=1)
            dec_dist = (z_spec - z_decoy).norm(dim=1)
            L_decoy = F.relu(pos_dist - dec_dist + self.decoy_margin).mean()

        loss_v_s = self.variance_loss(z_spec)
        loss_v_p = self.variance_loss(z_pep)
        loss_var = 2.0 * loss_v_s + 1.0 * loss_v_p

        loss = (L_align
                + self.lam * L_uniform
                + self.decoy_weight * L_decoy
                + self.lam_var * loss_var)

        with torch.no_grad():
            sim = z_spec @ z_pep.T
            labels = torch.arange(B, device=z_spec.device)
            acc = ((sim.argmax(dim=1) == labels).float().mean() +
                   (sim.argmax(dim=0) == labels).float().mean()) / 2

            diagnostics = {
                'L_align': L_align.item(),
                'L_uniform_spec': L_unif_s.item(),
                'L_uniform_pep': L_unif_p.item(),
                'loss_var': loss_var.item(),
                'pos_sim': (z_spec * z_pep).sum(dim=1).mean().item(),
            }
            if z_decoy is not None:
                pos_sim = (z_spec * z_pep).sum(dim=1).mean()
                dec_sim = (z_spec * z_decoy).sum(dim=1).mean()
                diagnostics['L_decoy'] = L_decoy.item()
                diagnostics['decoy_pos_sim'] = pos_sim.item()
                diagnostics['decoy_dec_sim'] = dec_sim.item()
                diagnostics['decoy_gap'] = (pos_sim - dec_sim).item()

        return loss, acc, diagnostics
