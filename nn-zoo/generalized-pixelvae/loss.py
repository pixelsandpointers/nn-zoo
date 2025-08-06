from typing import Dict

import torch
import torch.nn.functional as F


def vae_loss_function(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float,  # FIX: Added kl_weight parameter
    num_pixel_vals: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Calculates the VAE loss (ELBO).

    Args:
        logits (torch.Tensor): The output from the decoder. Shape: (B, C * num_pixel_vals, H, W)
        x (torch.Tensor): The ground truth input image. Shape: (B, C, H, W), values in [0, 1]
        mu (torch.Tensor): The latent mean.
        log_var (torch.Tensor): The latent log variance.
        kl_weight (float): The weight factor for KL.
        num_pixel_vals (int): The number of discrete values for each pixel.

    Returns:
        torch.Tensor: A scalar tensor representing the final loss.
    """
    batch_size = x.shape[0]

    # Reconstruction Loss
    logits_permuted = logits.view(
        batch_size, x.shape[1], num_pixel_vals, x.shape[2], x.shape[3]
    ).permute(0, 2, 1, 3, 4)
    logits_flat = logits_permuted.reshape(batch_size, num_pixel_vals, -1)
    x_labels = (x * (num_pixel_vals - 1)).long()
    x_labels_flat = x_labels.view(batch_size, -1)
    recon_loss = F.cross_entropy(logits_flat, x_labels_flat, reduction="sum")

    # KL Divergence
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # FIX: Apply the KL annealing weight to the KLD term
    total_loss = (recon_loss + kl_weight * kld) / batch_size

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss / batch_size,
        "kld": kld / batch_size,
    }
