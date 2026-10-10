from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    def __init__(self, initial_temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(initial_temperature)))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-4, max=100.0)

    def forward(self, rna_projection: torch.Tensor,
                protein_projection: torch.Tensor) -> torch.Tensor:
        rna_projection = F.normalize(rna_projection, dim=-1)
        protein_projection = F.normalize(protein_projection, dim=-1)
        logits = rna_projection @ protein_projection.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def zinb_negative_log_likelihood(counts: torch.Tensor, mean: torch.Tensor,
                                 dispersion: torch.Tensor, dropout_logits: torch.Tensor,
                                 eps: float = 1e-8) -> torch.Tensor:
    if dispersion.ndim == 1:
        dispersion = dispersion.unsqueeze(0)
    softplus_neg_pi = F.softplus(-dropout_logits)
    log_theta = torch.log(dispersion + eps)
    log_theta_mu = torch.log(dispersion + mean + eps)
    pi_theta_log = -dropout_logits + dispersion * (log_theta - log_theta_mu)
    zero_case = F.softplus(pi_theta_log) - softplus_neg_pi
    nonzero_case = (-softplus_neg_pi + pi_theta_log
                    + counts * (torch.log(mean + eps) - log_theta_mu)
                    + torch.lgamma(counts + dispersion) - torch.lgamma(dispersion)
                    - torch.lgamma(counts + 1.0))
    return -torch.where(counts < eps, zero_case, nonzero_case).mean()


@dataclass
class CorrectionLoss:
    total: torch.Tensor
    masked: torch.Tensor
    clean: torch.Tensor
    mean_mask_rate: float


def disagreement_mask_probability(disagreement: torch.Tensor, base: float,
                                  scale: float, maximum: float = 0.9) -> torch.Tensor:
    if not 0.0 <= base <= 1.0:
        raise ValueError("base mask probability must be in [0, 1].")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be in [0, 1].")
    relative = disagreement / (disagreement.mean(dim=-1, keepdim=True) + 1e-6)
    return (base * ((1.0 - scale) + scale * relative)).clamp(0.0, maximum)


def batch_protein_correlation_loss(prediction: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    covariance = (prediction * target).sum(dim=0)
    norms = prediction.norm(dim=0) * target.norm(dim=0) + 1e-6
    return 1.0 - (covariance / norms).mean()


def masked_residual_objective(model: nn.Module, prior: torch.Tensor, target: torch.Tensor,
                              query_latent: torch.Tensor, disagreement: torch.Tensor,
                              mask_probability: float, mask_disagreement_scale: float,
                              clean_weight: float, pcc_weight: float = 0.0) -> CorrectionLoss:
    rates = disagreement_mask_probability(disagreement, mask_probability,
                                          mask_disagreement_scale)
    keep = (torch.rand_like(prior) >= rates).to(prior.dtype)
    masked_prior = prior * keep

    masked_prediction = masked_prior + model(masked_prior, query_latent, disagreement)
    masked_loss = F.mse_loss(masked_prediction, target)

    clean_prediction = prior + model(prior, query_latent, disagreement)
    clean_loss = F.mse_loss(clean_prediction, target)
    total = masked_loss + clean_weight * clean_loss
    if pcc_weight:
        total = total + pcc_weight * batch_protein_correlation_loss(clean_prediction, target)

    return CorrectionLoss(
        total=total,
        masked=masked_loss, clean=clean_loss,
        mean_mask_rate=float(rates.mean().detach()),
    )


def flow_matching_objective(model: nn.Module, prior: torch.Tensor, target: torch.Tensor,
                            query_latent: torch.Tensor, confidence: torch.Tensor,
                            mask_probability: float, clean_weight: float) -> CorrectionLoss:
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1].")
    keep = (torch.rand_like(prior) >= mask_probability).to(prior.dtype)
    masked_prior = prior * keep

    t = torch.rand(prior.shape[0], device=prior.device, dtype=prior.dtype)
    x_t = (1.0 - t[:, None]) * masked_prior + t[:, None] * target
    flow_loss = F.mse_loss(model(x_t, t, query_latent, confidence), target - masked_prior)

    t0 = torch.zeros_like(t)
    endpoint_loss = F.mse_loss(prior + model(prior, t0, query_latent, confidence), target)

    return CorrectionLoss(total=flow_loss + clean_weight * endpoint_loss,
                          masked=flow_loss, clean=endpoint_loss,
                          mean_mask_rate=float(mask_probability))
