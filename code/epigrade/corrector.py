from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLNZeroBlock(nn.Module):

    def __init__(self, hidden_dim: int, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(context_dim, 3 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(context).chunk(3, dim=-1)
        transformed = self.mlp(self.norm(hidden) * (1.0 + scale) + shift)
        return hidden + torch.tanh(gate) * transformed


def _compress_disagreement(disagreement: torch.Tensor) -> torch.Tensor:
    compressed = torch.log1p(disagreement.clamp(min=0.0))
    return compressed / (compressed.mean(dim=-1, keepdim=True) + 1e-6)


class DisagreementGuidedCorrector(nn.Module):

    def __init__(self, protein_dim: int, rna_latent_dim: int = 128,
                 hidden_dim: int = 512, depth: int = 4, dropout: float = 0.1,
                 rna_skip_dim: int = 0) -> None:
        super().__init__()
        self.protein_dim = protein_dim
        self.rna_latent_dim = rna_latent_dim
        self.rna_skip_dim = rna_skip_dim
        if rna_skip_dim:
            self.register_buffer("gene_mean", torch.zeros(rna_skip_dim))
            self.register_buffer("gene_scale", torch.ones(rna_skip_dim))
            self.gene_projection = nn.Sequential(
                nn.Linear(rna_skip_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.input_projection = nn.Linear(protein_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(rna_latent_dim + protein_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, hidden_dim, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = nn.Linear(hidden_dim, protein_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def set_gene_scaling(self, genes) -> None:
        genes = torch.as_tensor(genes, dtype=self.gene_mean.dtype)
        self.gene_mean.copy_(genes.mean(dim=0))
        self.gene_scale.copy_(genes.std(dim=0).clamp(min=1e-3))

    def condition(self, query_rna_latent: torch.Tensor,
                  disagreement: torch.Tensor) -> torch.Tensor:
        latent = query_rna_latent[:, :self.rna_latent_dim]
        context = self.condition_projection(
            torch.cat([F.normalize(latent, dim=-1),
                       _compress_disagreement(disagreement)], dim=-1)
        )
        if self.rna_skip_dim:
            genes = query_rna_latent[:, self.rna_latent_dim:]
            context = context + self.gene_projection(
                ((genes - self.gene_mean) / self.gene_scale).clamp(-10.0, 10.0))
        return context

    def forward(self, protein_state: torch.Tensor, query_rna_latent: torch.Tensor,
                disagreement: torch.Tensor) -> torch.Tensor:
        context = self.condition(query_rna_latent, disagreement)
        hidden = self.input_projection(protein_state)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.residual_head(self.final_norm(hidden))

    @torch.no_grad()
    def predict(self, prior: torch.Tensor, query_rna_latent: torch.Tensor,
                disagreement: torch.Tensor) -> torch.Tensor:
        return prior + self(prior, query_rna_latent, disagreement)


def sinusoidal_time_embedding(t: torch.Tensor, dimension: int) -> torch.Tensor:
    t = t.reshape(-1)
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return F.pad(embedding, (0, 1)) if dimension % 2 else embedding


class FlowCorrector(nn.Module):

    def __init__(self, protein_dim: int, rna_latent_dim: int = 128,
                 hidden_dim: int = 512, depth: int = 4, time_dim: int = 64,
                 dropout: float = 0.1, condition_dim: int = 1) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.input_projection = nn.Linear(protein_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(rna_latent_dim + condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, hidden_dim, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, protein_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, protein_state: torch.Tensor, time: torch.Tensor,
                query_rna_latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition[:, None]
        if self.condition_dim > 1:
            condition = _compress_disagreement(condition)
        context = self.condition_projection(
            torch.cat([F.normalize(query_rna_latent, dim=-1), condition], dim=-1)
        ) + self.time_projection(
            sinusoidal_time_embedding(time.to(protein_state.dtype), self.time_dim)
        )
        hidden = self.input_projection(protein_state)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.velocity_head(self.final_norm(hidden))

    @torch.no_grad()
    def predict(self, prior: torch.Tensor, query_rna_latent: torch.Tensor,
                condition: torch.Tensor, steps: int = 1) -> torch.Tensor:
        state, step_size = prior, 1.0 / max(steps, 1)
        for index in range(steps):
            t = torch.full((state.shape[0],), index * step_size,
                           device=state.device, dtype=state.dtype)
            state = state + step_size * self(state, t, query_rna_latent, condition)
        return state
