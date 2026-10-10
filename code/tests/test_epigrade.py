from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.config import load_config
from epigrade.consensus import ReciprocalProteinMemory
from epigrade.corrector import DisagreementGuidedCorrector, FlowCorrector
from epigrade.objectives import (
    disagreement_mask_probability,
    flow_matching_objective,
    masked_residual_objective,
)


def _paired_data(n=500, latent=8, genes=40, proteins=12, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, latent)).astype(np.float32)
    rna = (z @ rng.normal(size=(latent, genes))).astype(np.float32)
    protein = (z @ rng.normal(size=(latent, proteins))
               + 0.2 * rng.normal(size=(n, proteins))).astype(np.float32)
    return rna, protein


def test_disagreement_is_per_protein_and_matches_weighted_spread():
    rna, protein = _paired_data()
    memory = ReciprocalProteinMemory(candidates=15, reference_degree=10)
    memory.fit(rna[:300], protein[:300])
    result = memory.retrieve(rna[300:], neighbors=5)

    assert result.disagreement.shape == (200, protein.shape[1])
    neighbours = protein[:300][result.indices[0]]
    weights = result.weights[0][:, None]
    expected = np.sqrt((weights * (neighbours - result.protein_prior[0]) ** 2).sum(0))
    assert np.allclose(result.disagreement[0], expected, atol=1e-5)


def test_disagreement_tracks_the_actual_error_of_the_prior():
    rna, protein = _paired_data(n=800, proteins=16, seed=3)
    memory = ReciprocalProteinMemory(candidates=20, reference_degree=15)
    memory.fit(rna[:500], protein[:500])
    result = memory.retrieve(rna[500:], neighbors=5)
    error = np.abs(result.protein_prior - protein[500:])
    per_protein = np.corrcoef(result.disagreement.mean(0), error.mean(0))[0, 1]
    assert per_protein > 0.5


def test_consistency_is_informative_where_cosine_saturates():
    rna, protein = _paired_data(n=600, seed=1)
    memory = ReciprocalProteinMemory(candidates=20, reference_degree=15)
    memory.fit(rna[:400], protein[:400])
    result = memory.retrieve(rna[400:], neighbors=5)
    assert 0.0 <= result.consistency.min() and result.consistency.max() <= 1.0
    assert result.consistency.std() > 0.01


def test_consistency_can_be_switched_off():
    rna, protein = _paired_data()
    plain = ReciprocalProteinMemory(use_consistency=False, candidates=15)
    plain.fit(rna[:300], protein[:300])
    result = plain.retrieve(rna[300:], neighbors=5)
    assert np.allclose(result.consistency, 1.0)


def test_spread_weights_do_not_collapse_when_similarities_saturate():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 16)).astype(np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    bank = (0.02 * features[:150] + np.array([1.0] + [0.0] * 15, dtype=np.float32))
    query = (0.02 * features[150:] + np.array([1.0] + [0.0] * 15, dtype=np.float32))
    protein = rng.normal(size=(150, 8)).astype(np.float32)

    spread = ReciprocalProteinMemory(use_spread_weights=True, use_consistency=False)
    spread.fit(bank, protein)
    raw = ReciprocalProteinMemory(use_spread_weights=False, use_consistency=False)
    raw.fit(bank, protein)
    w_spread = spread.retrieve(query, neighbors=5).weights
    w_raw = raw.retrieve(query, neighbors=5).weights
    assert w_spread.std(axis=1).mean() > w_raw.std(axis=1).mean()


def test_training_query_never_retrieves_its_own_label():
    rna, protein = _paired_data(n=300)
    memory = ReciprocalProteinMemory(candidates=15, reference_degree=10)
    memory.fit(rna, protein)
    result = memory.retrieve(rna, neighbors=5,
                             self_indices=np.arange(len(rna), dtype=np.int64))
    assert not (result.indices == np.arange(len(rna))[:, None]).any()


def test_mask_rate_is_aimed_but_preserves_the_expected_budget():
    disagreement = torch.rand(64, 20) * 3.0
    aimed = disagreement_mask_probability(disagreement, base=0.2, scale=1.0)
    uniform = disagreement_mask_probability(disagreement, base=0.2, scale=0.0)
    assert torch.allclose(uniform, torch.full_like(uniform, 0.2))
    assert abs(float(aimed.mean()) - 0.2) < 0.01
    assert float(aimed.std(dim=1).mean()) > 0.02


def test_mask_rate_follows_disagreement_ordering():
    disagreement = torch.tensor([[0.1, 1.0, 5.0]])
    rates = disagreement_mask_probability(disagreement, base=0.3, scale=1.0)
    assert rates[0, 0] < rates[0, 1] < rates[0, 2]


def test_mask_probability_arguments_are_validated():
    disagreement = torch.rand(4, 5)
    with pytest.raises(ValueError, match="base mask probability"):
        disagreement_mask_probability(disagreement, base=1.5, scale=0.5)
    with pytest.raises(ValueError, match="scale"):
        disagreement_mask_probability(disagreement, base=0.2, scale=2.0)


def test_corrector_starts_as_the_identity_on_the_prior():
    for model in (DisagreementGuidedCorrector(protein_dim=13),
                  FlowCorrector(protein_dim=13)):
        prior = torch.randn(6, 13)
        latent = torch.randn(6, 128)
        extra = torch.rand(6, 13) if isinstance(model, DisagreementGuidedCorrector) \
            else torch.rand(6, 1)
        assert torch.allclose(model.predict(prior, latent, extra), prior, atol=1e-6)


def test_corrector_actually_uses_the_disagreement_channel():
    torch.manual_seed(0)
    model = DisagreementGuidedCorrector(protein_dim=10)
    for block in model.blocks:
        torch.nn.init.normal_(block.modulation.weight, std=0.05)
    torch.nn.init.normal_(model.residual_head.weight, std=0.05)
    prior, latent = torch.randn(4, 10), torch.randn(4, 128)
    a = model(prior, latent, torch.ones(4, 10))
    b = model(prior, latent, torch.linspace(0.1, 3.0, 10).repeat(4, 1))
    assert not torch.allclose(a, b, atol=1e-4)


def test_compressed_disagreement_is_scale_free():
    from epigrade.corrector import _compress_disagreement
    base = torch.rand(8, 12) + 0.1
    assert torch.allclose(_compress_disagreement(base).mean(dim=-1),
                          torch.ones(8), atol=1e-4)


def test_residual_corrector_is_smaller_than_the_flow_ablation():
    residual = sum(p.numel() for p in DisagreementGuidedCorrector(protein_dim=50).parameters())
    flow = sum(p.numel() for p in FlowCorrector(protein_dim=50).parameters())
    assert residual < flow


def test_clean_term_is_evaluated_at_the_unmasked_prior():
    seen: list[torch.Tensor] = []

    class Recorder(torch.nn.Module):
        def forward(self, x, latent, disagreement):
            seen.append(x.detach().clone())
            return torch.zeros_like(x)

    torch.manual_seed(1)
    prior = torch.ones(4, 6)
    masked_residual_objective(Recorder(), prior, torch.zeros(4, 6), torch.zeros(4, 128),
                              torch.ones(4, 6), mask_probability=1.0,
                              mask_disagreement_scale=0.0, clean_weight=1.0)
    assert float((seen[0] == 0).float().mean()) > 0.7
    assert torch.allclose(seen[1], prior)


def test_flow_ablation_objective_still_runs():
    model = FlowCorrector(protein_dim=7)
    loss = flow_matching_objective(model, torch.randn(5, 7), torch.randn(5, 7),
                                   torch.randn(5, 128), torch.rand(5, 1),
                                   mask_probability=0.2, clean_weight=1.0)
    assert torch.isfinite(loss.total)


def test_default_configuration_validates_in_strict_mode():
    config = load_config(ROOT / "configs" / "epigrade.yaml", strict=True)
    assert config.method == "EpiGrade"
    assert config.correction.mask_disagreement_scale == 1.0
    assert not config.retrieval.use_consistency
    assert not config.retrieval.use_spread_weights


def test_training_priors_exclude_the_query_donor():
    import torch
    from dataclasses import replace as dc_replace

    from epigrade.aligner import CrossModalAligner
    from epigrade.training import retrieve_training_priors

    rng = np.random.default_rng(0)
    n_per_donor, n_donors, genes, proteins = 60, 5, 30, 7
    donors = np.repeat([f"P{i}" for i in range(n_donors)], n_per_donor)
    rna = rng.normal(size=(donors.size, genes)).astype(np.float32)
    protein = rng.normal(size=(donors.size, proteins)).astype(np.float32)

    config = load_config(ROOT / "configs" / "epigrade.yaml")
    config = dc_replace(config, retrieval=dc_replace(config.retrieval, neighbors=3),
                        correction=dc_replace(config.correction,
                                              training_group_exclusion=True))
    aligner = CrossModalAligner(rna_dim=genes, protein_dim=proteins,
                                latent_dim=config.alignment.latent_dim)
    result = retrieve_training_priors(aligner, rna, protein, config,
                                      torch.device("cpu"), groups=donors)

    assert result.indices.shape == (donors.size, 3)
    retrieved_donor = donors[result.indices]
    assert not (retrieved_donor == donors[:, None]).any()


def test_group_exclusion_falls_back_when_there_is_one_donor():
    import torch
    from dataclasses import replace as dc_replace

    from epigrade.aligner import CrossModalAligner
    from epigrade.training import retrieve_training_priors

    rng = np.random.default_rng(1)
    rna = rng.normal(size=(40, 12)).astype(np.float32)
    protein = rng.normal(size=(40, 5)).astype(np.float32)
    config = load_config(ROOT / "configs" / "epigrade.yaml")
    config = dc_replace(config, retrieval=dc_replace(config.retrieval, neighbors=3),
                        correction=dc_replace(config.correction,
                                              training_group_exclusion=True))
    aligner = CrossModalAligner(rna_dim=12, protein_dim=5,
                                latent_dim=config.alignment.latent_dim)
    result = retrieve_training_priors(aligner, rna, protein, config,
                                      torch.device("cpu"),
                                      groups=np.full(40, "only"))
    assert not (result.indices == np.arange(40)[:, None]).any()


def _reliability_fixture(n=400, genes=40, proteins=14, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 4)).astype(np.float32)
    rna = (latent @ rng.normal(size=(4, genes))).astype(np.float32)
    driven = latent @ rng.normal(size=(4, proteins // 2))
    noise = rng.normal(size=(n, proteins - proteins // 2))
    protein = np.hstack([driven, noise]).astype(np.float32)
    return rna, protein


def test_atlas_features_are_label_free_and_complete():
    from epigrade.reliability import FEATURES, atlas_features
    rna, protein = _reliability_fixture()
    types = np.repeat(["a", "b", "c", "d"], len(rna) // 4)
    features = atlas_features(rna, protein, cell_types=types,
                              disagreement=np.abs(protein).mean(axis=0))
    assert set(features) == set(FEATURES)
    for values in features.values():
        assert values.shape == (protein.shape[1],)
        assert np.isfinite(values).all()


def test_gene_correlation_separates_recoverable_proteins():
    from epigrade.reliability import atlas_features
    rna, protein = _reliability_fixture()
    half = protein.shape[1] // 2
    features = atlas_features(rna, protein)
    assert features["max_gene_corr"][:half].mean() > \
        features["max_gene_corr"][half:].mean() + 0.2


def test_estimator_predicts_unseen_proteins():
    from epigrade.reliability import ReliabilityEstimator, atlas_features
    rna, protein = _reliability_fixture(proteins=40)
    features = atlas_features(rna, protein)
    half = protein.shape[1] // 2
    observed = np.concatenate([np.full(half, 0.85), np.full(protein.shape[1] - half, 0.1)])

    train = np.r_[0:half // 2, half:half + (protein.shape[1] - half) // 2]
    test = np.setdiff1d(np.arange(protein.shape[1]), train)
    estimator = ReliabilityEstimator().fit(
        {k: v[train] for k, v in features.items()}, observed[train])
    predicted = estimator.predict({k: v[test] for k, v in features.items()})
    assert np.corrcoef(predicted, observed[test])[0, 1] > 0.7
    assert ((predicted >= 0.0) & (predicted <= 1.0)).all()


def test_assess_recommends_a_subpanel():
    from epigrade.reliability import ReliabilityEstimator, atlas_features
    rna, protein = _reliability_fixture(proteins=20)
    features = atlas_features(rna, protein)
    half = protein.shape[1] // 2
    observed = np.concatenate([np.full(half, 0.9), np.full(half, 0.05)])
    estimator = ReliabilityEstimator().fit(features, observed)
    panel = estimator.assess([f"P{i}" for i in range(protein.shape[1])],
                             features, threshold=0.5)
    assert 0 < len(panel.report()) < protein.shape[1]
    assert panel.recommended[:half].mean() > panel.recommended[half:].mean()


def test_rank_only_needs_no_reference_accuracies():
    from epigrade.reliability import atlas_features, rank_only
    rna, protein = _reliability_fixture(proteins=20)
    features = atlas_features(rna, protein,
                              disagreement=np.abs(protein).std(axis=0))
    scores = rank_only(features)
    assert scores.shape == (protein.shape[1],)
    assert 0.0 <= scores.min() and scores.max() <= 1.0
