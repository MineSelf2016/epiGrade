from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

__all__ = ["FusionSettings", "TranscriptomeNetwork", "Standardiser",
           "train_network", "fit_fusion", "apply_fusion", "fused_predictions"]


@dataclass(frozen=True)
class FusionSettings:
    members: int = 5
    hidden_dim: int = 512
    dropout: float = 0.25
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 500
    patience: int = 12
    decay_patience: int = 6
    decay_factor: float = 0.1
    min_improvement: float = 0.005
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.75, 0.9)
    gene_clip: float = 10.0


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float,
                 input_dropout: float | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if input_dropout is not None:
            layers += [nn.BatchNorm1d(in_dim), nn.Dropout(input_dropout)]
        layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.PReLU(),
                   nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TranscriptomeNetwork(nn.Module):

    def __init__(self, gene_dim: int, protein_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.25, n_quantiles: int = 4) -> None:
        super().__init__()
        self.input_block = _Block(gene_dim, hidden_dim, dropout, input_dropout=dropout)
        self.blocks = nn.ModuleList([_Block(hidden_dim, hidden_dim, dropout)
                                     for _ in range(3)])
        self.cell = nn.RNNCell(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, protein_dim)
        self.quantile_head = nn.Linear(hidden_dim, protein_dim * n_quantiles)
        self.n_quantiles = n_quantiles

    def forward(self, genes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_block(genes)
        state = self.cell(x, torch.zeros_like(x))
        for block in self.blocks:
            x = block(x)
            state = self.cell(x, state)
        quantiles = self.quantile_head(state).view(len(state), -1, self.n_quantiles)
        return self.mean_head(state), quantiles


@dataclass
class Standardiser:

    mean: np.ndarray
    scale: np.ndarray
    clip: float | None = None

    @classmethod
    def fit(cls, values: np.ndarray, clip: float | None = None) -> "Standardiser":
        values = np.asarray(values, dtype=np.float64)
        return cls(values.mean(axis=0).astype(np.float32),
                   (values.std(axis=0) + 1e-3).astype(np.float32), clip)

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = (np.asarray(values, np.float32) - self.mean) / self.scale
        return (out.clip(-self.clip, self.clip) if self.clip else out).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, np.float32) * self.scale + self.mean).astype(np.float32)


def train_network(fit_genes: np.ndarray, fit_protein: np.ndarray,
                  validation_genes: np.ndarray, validation_protein: np.ndarray,
                  predict_genes: list[np.ndarray], settings: FusionSettings,
                  device: torch.device, seed: int) -> tuple[list[np.ndarray], int]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    x_fit, y_fit = tensor(fit_genes), tensor(fit_protein)
    x_val, y_val = tensor(validation_genes), tensor(validation_protein)
    levels = torch.tensor(settings.quantiles, device=device)

    network = TranscriptomeNetwork(x_fit.shape[1], y_fit.shape[1], settings.hidden_dim,
                                   settings.dropout, len(settings.quantiles)).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)

    best, best_state, best_epoch = np.inf, None, 0
    stale = stale_lr = 0
    for epoch in range(1, settings.max_epochs + 1):
        network.train()
        order = torch.randperm(len(x_fit), device=device, generator=generator)
        for start in range(0, len(order), settings.batch_size):
            batch = order[start:start + settings.batch_size]
            if len(batch) < 2:
                continue
            mean, quantiles = network(x_fit[batch])
            target = y_fit[batch]
            error = target[:, :, None] - quantiles
            pinball = torch.maximum(levels * error, (levels - 1.0) * error).mean()
            loss = ((mean - target) ** 2).mean() + pinball
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            validation_loss = float(((network(x_val)[0] - y_val) ** 2).mean())
        if validation_loss < best / (1.0 + settings.min_improvement):
            best, best_epoch, stale, stale_lr = validation_loss, epoch, 0, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            stale_lr += 1
            if stale_lr >= settings.decay_patience:
                for group in optimizer.param_groups:
                    group["lr"] *= settings.decay_factor
                stale_lr = 0
            if stale >= settings.patience:
                break

    network.load_state_dict(best_state)
    network.eval()
    outputs = []
    with torch.no_grad():
        for genes in predict_genes:
            chunks = [network(tensor(genes[i:i + 8192]))[0].cpu().numpy()
                      for i in range(0, len(genes), 8192)]
            outputs.append(np.concatenate(chunks, axis=0))
    return outputs, best_epoch


def fit_fusion(sources: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    n_cells, n_proteins = target.shape
    weights = np.empty((n_proteins, len(sources) + 1), dtype=np.float64)
    ones = np.ones(n_cells)
    for p in range(n_proteins):
        design = np.stack([s[:, p] for s in sources] + [ones], axis=1)
        weights[p] = np.linalg.lstsq(design, target[:, p], rcond=None)[0]
    return weights


def apply_fusion(weights: np.ndarray, sources: list[np.ndarray]) -> np.ndarray:
    out = weights[:, -1][None, :].astype(np.float64)
    for index, source in enumerate(sources):
        out = out + weights[:, index][None, :] * source
    return out.astype(np.float32)


@dataclass
class FusionResult:
    predictions: dict[str, np.ndarray]
    fusion_weights: np.ndarray
    single_fusion_weights: np.ndarray
    best_epochs: list[int] = field(default_factory=list)


def fused_predictions(fit_genes: np.ndarray, fit_protein: np.ndarray,
                      validation_genes: np.ndarray, validation_protein: np.ndarray,
                      query_genes: np.ndarray, validation_prior: np.ndarray,
                      query_prior: np.ndarray, settings: FusionSettings,
                      device: torch.device, seed: int) -> FusionResult:
    genes = Standardiser.fit(fit_genes, clip=settings.gene_clip)
    protein = Standardiser.fit(fit_protein)
    z_fit, z_val, z_query = (genes.transform(a) for a in
                             (fit_genes, validation_genes, query_genes))
    y_fit, y_val = protein.transform(fit_protein), protein.transform(validation_protein)

    val_members, query_members, epochs = [], [], []
    for member in range(settings.members):
        (val_pred, query_pred), epoch = train_network(
            z_fit, y_fit, z_val, y_val, [z_val, z_query], settings, device,
            seed=seed + 1000 * member)
        val_members.append(protein.inverse(val_pred))
        query_members.append(protein.inverse(query_pred))
        epochs.append(epoch)

    val_mean, query_mean = np.mean(val_members, axis=0), np.mean(query_members, axis=0)
    weights = fit_fusion([val_mean, validation_prior], validation_protein)
    single = fit_fusion([val_members[0], validation_prior], validation_protein)
    predictions = {
        "full": apply_fusion(weights, [query_mean, query_prior]),
        "ensemble": query_mean.astype(np.float32),
        "network": query_members[0].astype(np.float32),
        "network_fused": apply_fusion(single, [query_members[0], query_prior]),
        "average": (0.5 * (query_mean + query_prior)).astype(np.float32),
    }
    return FusionResult(predictions, weights, single, epochs)
