from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CHECKED = ("1.1.6",)
REFERENCE, QUERY = 0, 1


def main(payload_path: str, output_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch

    warnings.resetwarnings()
    warnings.simplefilter("ignore")

    if scvi.__version__ not in CHECKED:
        raise SystemExit(
            f"scvi-tools {scvi.__version__} has not been checked for this "
            f"worker; {CHECKED[0]} reconstructs training-set protein and "
            "1.4.0 does not. Pin the environment before trusting the output.")

    payload = np.load(payload_path, allow_pickle=True)
    reference_rna = payload["reference_rna"]
    reference_protein = payload["reference_protein"]
    query_rna = payload["query_rna"]
    n_reference, n_query = len(reference_rna), len(query_rna)
    n_proteins = reference_protein.shape[1]
    scvi.settings.seed = int(payload["seed"]) if "seed" in payload else 0

    non_integer = float(np.mean(~np.isclose(reference_protein,
                                            np.round(reference_protein))))
    if non_integer > 0:
        raise SystemExit(
            f"PRECONDITION: {100 * non_integer:.2f}% of protein entries are not "
            "integers after clone harmonisation; totalVI requires unnormalised "
            "counts, so no number is reported for this panel")

    counts = np.vstack([reference_rna, query_rna]).astype(np.float32)
    if "n_hvg" in payload:
        wanted = min(int(payload["n_hvg"]), counts.shape[1])
        variance = counts[:n_reference].var(axis=0)
        keep = np.sort(np.argsort(variance)[::-1][:wanted])
        counts = counts[:, keep]

    protein = np.vstack([
        reference_protein,
        np.zeros((n_query, n_proteins), dtype=np.float32)]).astype(np.float32)

    data = ad.AnnData(
        counts,
        obs=pd.DataFrame(
            {"batch_index": pd.Categorical([REFERENCE] * n_reference
                                           + [QUERY] * n_query)},
            index=[f"c{i}" for i in range(n_reference + n_query)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(counts.shape[1])]))
    data.obsm["protein_expression"] = pd.DataFrame(
        protein, index=data.obs_names,
        columns=[f"p{i}" for i in range(n_proteins)])

    scvi.model.TOTALVI.setup_anndata(
        data, batch_key="batch_index",
        protein_expression_obsm_key="protein_expression")
    model = scvi.model.TOTALVI(data, latent_distribution="normal",
                               n_layers_decoder=2)
    try:
        model.train(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    except RuntimeError as error:
        print(f"GPU unavailable ({error}); falling back to CPU", flush=True)
        model.train(accelerator="cpu")

    _, predicted = model.get_normalized_expression(
        n_samples=25, transform_batch=REFERENCE, include_protein_background=True,
        sample_protein_mixing=False, return_mean=True)
    frame = predicted if hasattr(predicted, "columns") else None
    values = (frame.reindex(columns=[f"p{i}" for i in range(n_proteins)]).to_numpy()
              if frame is not None else np.asarray(predicted))
    prediction = np.asarray(values[n_reference:], dtype=np.float32)

    trained = np.asarray(values[:n_reference], dtype=np.float32)
    observed = reference_protein
    fidelity = float(np.nanmean([
        np.corrcoef(trained[:, i], observed[:, i])[0, 1]
        for i in range(n_proteins)]))
    print(f"totalVI training-set reconstruction PCC-P = {fidelity:.4f}",
          flush=True)
    import json as _json
    (Path(output_path).parent / "diagnostics.json").write_text(
        _json.dumps({"training_reconstruction_pcc_p": fidelity}))
    if fidelity < 0.2:
        raise SystemExit(
            f"totalVI reconstructs its own training protein at {fidelity:.4f}; "
            "the run is not usable and no number should be taken from it")

    if prediction.shape != (n_query, n_proteins):
        raise SystemExit(
            f"totalVI returned {prediction.shape}, expected {(n_query, n_proteins)}")
    np.save(output_path, np.nan_to_num(prediction, nan=0.0,
                                       posinf=0.0, neginf=0.0))
    print(f"totalVI wrote {prediction.shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
