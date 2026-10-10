from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epigrade.panels import canonicalise_panel, collapse_duplicates


def _read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _attach_proteins(adata: ad.AnnData, protein: np.ndarray, names: list[str]) -> None:
    _, groups = canonicalise_panel(names)
    collapsed, symbols = collapse_duplicates(np.asarray(protein, dtype=np.float32), groups)
    adata.obsm["protein_expression"] = pd.DataFrame(
        collapsed, index=adata.obs_names, columns=symbols
    )
    adata.uns["raw_protein_names"] = list(map(str, names))


def _finalise(adata: ad.AnnData, out: Path, label: str) -> None:
    adata.X = sp.csr_matrix(adata.X) if not sp.issparse(adata.X) else adata.X.tocsr()
    adata.X.eliminate_zeros()
    adata.var_names_make_unique()
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    protein = adata.obsm["protein_expression"]
    print(
        f"[{label}] cells={adata.n_obs} genes={adata.n_vars} "
        f"proteins={protein.shape[1]} -> {out}"
    )
    for key in ("donor", "batch", "time", "celltype_l1"):
        if key in adata.obs:
            print(f"    {key}: {adata.obs[key].nunique()} levels")


def prepare_cbmc(raw: Path, out: Path) -> None:
    rna = pd.read_csv(raw / "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz", index_col=0)
    adt = pd.read_csv(raw / "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz", index_col=0)
    adt = adt[rna.columns]

    human = [g for g in rna.index if str(g).startswith("HUMAN_")]
    rna = rna.loc[human]
    rna.index = [g.replace("HUMAN_", "", 1) for g in rna.index]

    adata = ad.AnnData(
        X=sp.csr_matrix(rna.to_numpy(dtype=np.float32).T),
        obs=pd.DataFrame(index=pd.Index(rna.columns.astype(str), name=None)),
        var=pd.DataFrame(index=pd.Index(rna.index.astype(str), name=None)),
    )
    adata.obs["donor"] = "CBMC_pool"
    adata.obs["batch"] = "GSE100866"
    _attach_proteins(adata, adt.to_numpy(dtype=np.float32).T, list(adt.index.astype(str)))
    _finalise(adata, out, "cbmc")


def prepare_hao(raw: Path, meta_dir: Path, assay: str, out: Path) -> None:
    prefix, rna_gsm, adt_gsm = {
        "3p": ("3P", "GSM5008737", "GSM5008738"),
        "5p": ("5P", "GSM5008740", "GSM5008741"),
    }[assay]

    rna = sio.mmread(raw / f"{rna_gsm}_RNA_{prefix}-matrix.mtx.gz").tocsr()
    genes = [l.split("\t")[0] for l in _read_lines(raw / f"{rna_gsm}_RNA_{prefix}-features.tsv.gz")]
    barcodes = _read_lines(raw / f"{rna_gsm}_RNA_{prefix}-barcodes.tsv.gz")

    adt = sio.mmread(raw / f"{adt_gsm}_ADT_{prefix}-matrix.mtx.gz").tocsr()
    proteins = [l.split("\t")[0] for l in _read_lines(raw / f"{adt_gsm}_ADT_{prefix}-features.tsv.gz")]
    adt_barcodes = _read_lines(raw / f"{adt_gsm}_ADT_{prefix}-barcodes.tsv.gz")
    if adt_barcodes != barcodes:
        order = pd.Index(adt_barcodes).get_indexer(pd.Index(barcodes))
        if (order < 0).any():
            raise ValueError("RNA and ADT barcodes cannot be aligned.")
        adt = adt[:, order]

    adata = ad.AnnData(
        X=rna.T.tocsr().astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes)),
        var=pd.DataFrame(index=pd.Index(genes)),
    )

    meta = pd.read_csv(meta_dir / f"GSE164378_sc.meta.data_{prefix}.csv.gz", index_col=0)
    meta = meta.reindex(adata.obs_names)
    for source, target in (
        ("donor", "donor"), ("Batch", "batch"), ("time", "time"), ("lane", "lane"),
        ("celltype.l1", "celltype_l1"), ("celltype.l2", "celltype_l2"),
        ("celltype.l3", "celltype_l3"),
    ):
        if source in meta:
            adata.obs[target] = meta[source].astype(str).values

    _attach_proteins(adata, np.asarray(adt.T.todense(), dtype=np.float32), proteins)
    _finalise(adata, out, f"hao_{assay}")


def prepare_covid(raw: Path, out: Path) -> None:
    features = [l.split("\t") for l in _read_lines(raw / "GSE155673_features.tsv.gz")]
    gene_rows = [i for i, f in enumerate(features) if len(f) > 2 and f[2] == "Gene Expression"]
    prot_rows = [i for i, f in enumerate(features) if len(f) > 2 and f[2] != "Gene Expression"]
    genes = [features[i][1] for i in gene_rows]
    proteins = [features[i][1] for i in prot_rows]

    donors = sorted({p.name.split("_")[1] for p in raw.glob("GSE155673_cov*_matrix.mtx.gz")})
    rna_blocks, adt_blocks, obs_rows = [], [], []
    for donor in donors:
        matrix = sio.mmread(raw / f"GSE155673_{donor}_matrix.mtx.gz").tocsr()
        barcodes = _read_lines(raw / f"GSE155673_{donor}_barcodes.tsv.gz")
        if matrix.shape[1] != len(barcodes):
            raise ValueError(f"{donor}: barcode count does not match the matrix.")
        rna_blocks.append(matrix[gene_rows].T.tocsr().astype(np.float32))
        adt_blocks.append(np.asarray(matrix[prot_rows].T.todense(), dtype=np.float32))
        obs_rows.append(pd.DataFrame({"donor": donor}, index=[f"{donor}_{b}" for b in barcodes]))
        print(f"    {donor}: {len(barcodes)} cells")

    adata = ad.AnnData(
        X=sp.vstack(rna_blocks).tocsr(),
        obs=pd.concat(obs_rows),
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    adata.obs["batch"] = "GSE155673"
    adata.obs["condition"] = "COVID-19"
    _attach_proteins(adata, np.vstack(adt_blocks), proteins)
    _finalise(adata, out, "covid_pbmc")


def prepare_bonemarrow(raw: Path, out: Path) -> None:
    rna = pd.read_csv(raw / "GSM3681518_MNC_RNA_counts.tsv.gz", sep="\t", index_col=0)
    adt = pd.read_csv(raw / "GSM3681519_MNC_ADT_counts.tsv.gz", sep="\t", index_col=0)
    shared = [c for c in rna.columns if c in set(adt.columns)]
    rna, adt = rna[shared], adt[shared]

    adata = ad.AnnData(
        X=sp.csr_matrix(rna.to_numpy(dtype=np.float32).T),
        obs=pd.DataFrame(index=pd.Index(shared)),
        var=pd.DataFrame(index=pd.Index(rna.index.astype(str))),
    )
    adata.obs["donor"] = [c.split("_")[0] for c in shared]
    adata.obs["batch"] = "GSE128639"
    adata.obs["tissue"] = "bone_marrow"
    _attach_proteins(adata, adt.to_numpy(dtype=np.float32).T, list(adt.index.astype(str)))
    _finalise(adata, out, "bonemarrow")


DATASETS = ("cbmc", "eccite_5p", "gse3p_3p", "covid_pbmc", "bonemarrow")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS + ("all",), default="all")
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--raw-root", type=Path, default=root / "data" / "raw")
    parser.add_argument("--out-root", type=Path, default=root / "data" / "processed")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in targets:
        out = args.out_root / f"{name}.h5ad"
        if out.exists():
            print(f"[{name}] exists, skipping")
            continue
        if name == "cbmc":
            prepare_cbmc(args.raw_root / "cbmc", out)
        elif name == "eccite_5p":
            prepare_hao(args.raw_root / "eccite_5p", args.raw_root / "meta", "5p", out)
        elif name == "gse3p_3p":
            prepare_hao(args.raw_root / "gse3p_3p", args.raw_root / "meta", "3p", out)
        elif name == "covid_pbmc":
            prepare_covid(args.raw_root / "covid_pbmc", out)
        elif name == "bonemarrow":
            prepare_bonemarrow(args.raw_root / "bonemarrow", out)


if __name__ == "__main__":
    main()
