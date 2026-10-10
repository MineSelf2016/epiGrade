from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .objectives import SymmetricInfoNCE, zinb_negative_log_likelihood


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.gate(self.norm(x)).chunk(2, dim=-1)
        return x + self.output(self.dropout(F.gelu(value * torch.sigmoid(gate))))


class GatedResidualEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 depth: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x)
        for block in self.blocks:
            h = block(h)
        return self.output_projection(self.output_norm(h))


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.dropout_head = nn.Linear(hidden_dim, output_dim)
        self.dispersion = nn.Parameter(torch.zeros(output_dim))

    def forward(self, latent: torch.Tensor):
        hidden = self.backbone(latent)
        return (F.softplus(self.mean_head(hidden)) + 1e-4,
                F.softplus(self.dispersion) + 1e-4,
                self.dropout_head(hidden))


class CrossModalAligner(nn.Module):

    def __init__(self, rna_dim: int, protein_dim: int, hidden_dim: int = 256,
                 latent_dim: int = 128, projection_dim: int = 64, depth: int = 2,
                 dropout: float = 0.1, temperature_init: float = 0.07) -> None:
        super().__init__()
        self.rna_encoder = GatedResidualEncoder(rna_dim, hidden_dim, latent_dim, depth, dropout)
        self.protein_encoder = GatedResidualEncoder(protein_dim, hidden_dim, latent_dim, depth, dropout)
        self.rna_projection = ProjectionHead(latent_dim, projection_dim)
        self.protein_projection = ProjectionHead(latent_dim, projection_dim)
        self.rna_decoder = ZINBDecoder(latent_dim, hidden_dim, rna_dim)
        self.protein_decoder = ZINBDecoder(latent_dim, hidden_dim, protein_dim)
        self.info_nce = SymmetricInfoNCE(temperature_init)

    def encode_rna(self, rna_normalized: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(rna_normalized)

    def forward(self, rna_normalized: torch.Tensor, protein_normalized: torch.Tensor):
        rna_latent = self.rna_encoder(rna_normalized)
        protein_latent = self.protein_encoder(protein_normalized)
        rna_mean, rna_dispersion, rna_dropout = self.rna_decoder(rna_latent)
        protein_mean, protein_dispersion, protein_dropout = self.protein_decoder(protein_latent)
        return {
            "rna_latent": rna_latent, "protein_latent": protein_latent,
            "rna_projection": self.rna_projection(rna_latent),
            "protein_projection": self.protein_projection(protein_latent),
            "rna_mean": rna_mean, "rna_dispersion": rna_dispersion, "rna_dropout": rna_dropout,
            "protein_mean": protein_mean, "protein_dispersion": protein_dispersion,
            "protein_dropout": protein_dropout,
        }

    def objective(self, rna_normalized, rna_counts, protein_normalized, protein_counts,
                  contrastive_weight, rna_zinb_weight, protein_zinb_weight):
        output = self(rna_normalized, protein_normalized)
        contrastive = self.info_nce(output["rna_projection"], output["protein_projection"])
        rna_zinb = zinb_negative_log_likelihood(
            rna_counts, output["rna_mean"], output["rna_dispersion"], output["rna_dropout"])
        protein_zinb = zinb_negative_log_likelihood(
            protein_counts, output["protein_mean"], output["protein_dispersion"],
            output["protein_dropout"])
        return {
            "total": contrastive_weight * contrastive + rna_zinb_weight * rna_zinb
                     + protein_zinb_weight * protein_zinb,
            "contrastive": contrastive, "rna_zinb": rna_zinb, "protein_zinb": protein_zinb,
        }
