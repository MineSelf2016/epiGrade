from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(payload_path: str, output_path: str) -> None:
    import anndata as ad
    import pandas as pd
    from sciPENN.sciPENN_API import sciPENN_API

    import torch

    payload = np.load(payload_path, allow_pickle=True)
    seed = int(payload["seed"]) if "seed" in payload else 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reference_rna = payload["reference_rna"]
    reference_protein = payload["reference_protein"]
    query_rna = payload["query_rna"]
    n_proteins = reference_protein.shape[1]

    genes = [f"g{i}" for i in range(reference_rna.shape[1])]
    proteins = [f"p{i}" for i in range(n_proteins)]

    train_gene = ad.AnnData(reference_rna.astype(np.float32),
                            var=pd.DataFrame(index=genes))
    train_protein = ad.AnnData(reference_protein.astype(np.float32),
                               var=pd.DataFrame(index=proteins))
    test_gene = ad.AnnData(query_rna.astype(np.float32),
                           var=pd.DataFrame(index=genes))

    model = sciPENN_API(
        gene_trainsets=[train_gene], protein_trainsets=[train_protein],
        gene_test=test_gene, select_hvg=False,
        min_cells=0, min_genes=0, use_gpu=True)
    model.train(n_epochs=int(payload["n_epochs"]) if "n_epochs" in payload else 500,
                ES_max=12, decay_max=6, decay_step=0.1, lr=1e-3,
                weights_dir="scipenn_weights", load=False)
    imputed = model.predict()

    prediction = np.asarray(imputed.X, dtype=np.float32)
    if prediction.shape[0] != query_rna.shape[0]:
        raise SystemExit(
            f"sciPENN returned {prediction.shape[0]} rows for "
            f"{query_rna.shape[0]} query cells; filtering must stay disabled")
    if prediction.shape[1] != n_proteins:
        raise SystemExit(
            f"sciPENN returned {prediction.shape[1]} proteins, expected {n_proteins}")
    np.save(output_path, np.nan_to_num(prediction, nan=0.0,
                                       posinf=0.0, neginf=0.0))
    print(f"sciPENN wrote {prediction.shape}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
