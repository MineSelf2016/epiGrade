from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.data import generate_grouped_splits, validate_splits
from epigrade.gating import otsu_threshold, quadrant_fractions
from epigrade.metrics import (
    central_moment_discrepancy,
    correlation_matrix_distance,
    rbf_mmd,
)
from epigrade.panels import canonical_protein, canonicalise_panel, collapse_duplicates
from epigrade.stats import (
    cluster_bootstrap_ci,
    corrected_resampled_ttest,
    holm_bonferroni,
    paired_ttest,
)


@pytest.mark.parametrize("raw,expected", [
    ("CD3--UCHT1-TSA", "CD3"),
    ("CD279_PD-1--RMP1-30-TSA", "CD279"),
    ("PD-1(CD279)", "CD279"),
    ("IL7Ralpha(CD127)", "CD127"),
    ("MHCII(HLA-DR)", "HLA-DR"),
    ("HLA.DR", "HLA-DR"),
    ("CD45RA-Bio", "CD45RA"),
    ("CD4-1", "CD4"),
    ("CD8a", "CD8"),
    ("CCR7", "CD197"),
    ("TCRa7-2--3C10-TSA", "TCR-VA7.2"),
    ("TCR-V-7.2", "TCR-VA7.2"),
])
def test_canonical_protein(raw, expected):
    assert canonical_protein(raw) == expected


@pytest.mark.parametrize("control", [
    "Isotype-control--MOPC-21-TSA", "Rat-IgG1-1", "Rat-IgG2b", "IgG2a",
    "Anti-PE--PE001-TSA",
])
def test_controls_are_dropped(control):
    assert canonical_protein(control) is None


def test_clone_index_strip_does_not_damage_non_cd_names():
    assert canonical_protein("TCRa7-2") == "TCR-VA7.2"


def test_replicate_antibodies_are_averaged_not_summed():
    names = ["CD45RA", "CD45RA-Bio", "CD3"]
    matrix = np.array([[10.0, 20.0, 5.0]], dtype=np.float32)
    _, groups = canonicalise_panel(names)
    collapsed, symbols = collapse_duplicates(matrix, groups)
    assert symbols == ["CD3", "CD45RA"]
    assert collapsed[0, symbols.index("CD45RA")] == pytest.approx(15.0)


def test_correlation_matrix_distance_is_zero_for_identical_inputs():
    rng = np.random.default_rng(0)
    matrix = np.corrcoef(rng.normal(size=(20, 200)))
    assert correlation_matrix_distance(matrix, matrix) == pytest.approx(0.0, abs=1e-9)


def test_central_moment_discrepancy_is_zero_for_identical_inputs():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 8))
    assert central_moment_discrepancy(values, values) == pytest.approx(0.0, abs=1e-9)


def test_mmd_does_not_depend_on_panel_size():
    rng = np.random.default_rng(0)
    values = [rbf_mmd(rng.normal(size=(600, d)), rng.normal(size=(600, d)))
              for d in (13, 48, 211)]
    assert max(values) < 0.02
    assert max(values) - min(values) < 0.01


def test_mmd_detects_a_real_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(600, 20))
    assert rbf_mmd(a, a + 3.0) > rbf_mmd(a, rng.normal(size=(600, 20)))


def test_nadeau_bengio_inflates_the_standard_error_by_exactly_1_5():
    differences = np.array([0.017, 0.015, 0.019, 0.014, 0.018])
    naive = paired_ttest(differences)
    corrected = corrected_resampled_ttest(differences, n_train=8000, n_test=2000)
    assert naive.statistic / corrected.statistic == pytest.approx(1.5, rel=1e-6)
    assert corrected.pvalue > naive.pvalue


def test_holm_is_monotone_and_never_below_the_raw_value():
    differences = [np.array([0.05] * 5), np.array([0.01] * 5), np.array([0.002] * 5)]
    results = [corrected_resampled_ttest(d, 8000, 2000, comparison=str(i))
               for i, d in enumerate(differences)]
    adjusted = holm_bonferroni(results)
    values = [row["pvalue_holm"] for row in adjusted]
    for row in adjusted:
        assert row["pvalue_holm"] >= row["pvalue"] - 1e-12
    ordered = [v for _, v in sorted(zip([r.pvalue for r in results], values))]
    assert ordered == sorted(ordered)


def test_cluster_bootstrap_is_wider_than_the_cell_level_interval():
    from epigrade.metrics import protein_pearson_mean
    rng = np.random.default_rng(0)
    donors = np.repeat([f"P{i}" for i in range(6)], 300)
    offsets = {d: rng.normal(scale=0.7) for d in np.unique(donors)}
    target = rng.normal(size=(donors.size, 10))
    prediction = target + np.array([[offsets[d]] for d in donors]) \
        + 0.4 * rng.normal(size=target.shape)
    grouped = cluster_bootstrap_ci(prediction, target, donors, protein_pearson_mean,
                                   n_boot=200)
    from epigrade.stats import cell_bootstrap_ci
    cells = cell_bootstrap_ci(prediction, target, protein_pearson_mean, n_boot=200)
    assert (grouped["ci_upper"] - grouped["ci_lower"]) > (cells["ci_upper"] - cells["ci_lower"])


def test_cluster_bootstrap_refuses_a_single_group():
    from epigrade.metrics import protein_pearson_mean
    with pytest.raises(ValueError, match="at least two groups"):
        cluster_bootstrap_ci(np.zeros((10, 3)), np.zeros((10, 3)),
                             np.zeros(10), protein_pearson_mean, n_boot=10)


def test_grouped_splits_never_leak_a_donor_into_its_own_test_fold():
    groups = np.repeat([f"P{i}" for i in range(8)], 120)
    splits = generate_grouped_splits(groups, folds=5, seed=42,
                                     validation_fraction_of_reference=0.125)
    validate_splits(splits, groups.size, 5)
    for split in splits:
        reference = set(groups[np.asarray(split["fit"])]) | \
            set(groups[np.asarray(split["validation"])])
        assert not reference & set(groups[np.asarray(split["test"])])


def test_grouped_splits_require_enough_groups():
    with pytest.raises(ValueError, match="at least 5"):
        generate_grouped_splits(np.repeat(["a", "b"], 10), folds=5, seed=0,
                                validation_fraction_of_reference=0.125)


def test_otsu_lands_between_two_separated_modes():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 0.3, 4000),
                             rng.normal(5.0, 0.3, 4000)])
    assert 1.0 < otsu_threshold(values) < 4.0


def test_otsu_ignores_non_finite_values():
    rng = np.random.default_rng(1)
    clean = np.concatenate([rng.normal(0.0, 0.2, 2000),
                            rng.normal(4.0, 0.2, 2000)])
    dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
    assert otsu_threshold(dirty) == pytest.approx(otsu_threshold(clean), abs=0.1)


def test_otsu_rejects_input_with_no_finite_values():
    with pytest.raises(ValueError):
        otsu_threshold(np.array([np.nan, np.inf]))


def test_quadrants_partition_the_events():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=500), rng.normal(size=500)
    parts = quadrant_fractions(x, y, 0.0, 0.0)
    assert sum(parts.values()) == pytest.approx(1.0)
    assert set(parts) == {"++", "+-", "-+", "--"}


def test_quadrants_place_events_on_the_expected_side():
    x = np.array([1.0, 1.0, -1.0, -1.0])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    assert quadrant_fractions(x, y, 0.0, 0.0) == {
        "++": 0.25, "+-": 0.25, "-+": 0.25, "--": 0.25}


def test_quadrants_require_matching_lengths():
    with pytest.raises(ValueError):
        quadrant_fractions(np.zeros(3), np.zeros(4), 0.0, 0.0)


def test_gate_boundary_counts_as_negative():
    parts = quadrant_fractions(np.array([0.0]), np.array([0.0]), 0.0, 0.0)
    assert parts["--"] == 1.0
