suppressPackageStartupMessages({
  library(jsonlite)
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(args[1])
out_path <- if (length(args) >= 2) args[2] else spec$output

read_matrix <- function(name) {
  shape <- spec$shapes[[name]]
  values <- readBin(file.path(spec$dir, paste0(name, ".bin")),
                    what = "numeric", size = 4, n = shape[1] * shape[2])
  matrix(values, nrow = shape[1], ncol = shape[2], byrow = TRUE)
}

reference_rna <- read_matrix("reference_rna")
reference_protein <- read_matrix("reference_protein")
query_rna <- read_matrix("query_rna")

n_genes <- ncol(reference_rna)
n_proteins <- ncol(reference_protein)
gene_names <- paste0("g", seq_len(n_genes))
protein_names <- paste0("p", seq_len(n_proteins))

ref_counts <- t(reference_rna)
rownames(ref_counts) <- gene_names
colnames(ref_counts) <- paste0("ref", seq_len(nrow(reference_rna)))
qry_counts <- t(query_rna)
rownames(qry_counts) <- gene_names
colnames(qry_counts) <- paste0("qry", seq_len(nrow(query_rna)))
adt_counts <- t(reference_protein)
rownames(adt_counts) <- protein_names
colnames(adt_counts) <- colnames(ref_counts)

reference <- CreateSeuratObject(counts = as(ref_counts, "dgCMatrix"))
reference[["ADT"]] <- CreateAssayObject(counts = as(adt_counts, "dgCMatrix"))
query <- CreateSeuratObject(counts = as(qry_counts, "dgCMatrix"))

n_features <- min(2000L, n_genes)
n_dims <- min(30L, n_genes - 1L, ncol(reference) - 1L, ncol(query) - 1L)

reference <- NormalizeData(reference, verbose = FALSE)
reference <- FindVariableFeatures(reference, nfeatures = n_features,
                                  verbose = FALSE)
reference <- ScaleData(reference, verbose = FALSE)
reference <- RunPCA(reference, npcs = n_dims, verbose = FALSE)
DefaultAssay(reference) <- "ADT"
reference <- NormalizeData(reference, normalization.method = "LogNormalize",
                           verbose = FALSE)
DefaultAssay(reference) <- "RNA"

query <- NormalizeData(query, verbose = FALSE)

anchors <- FindTransferAnchors(
  reference = reference, query = query,
  normalization.method = "LogNormalize", reference.reduction = "pca",
  dims = seq_len(n_dims), k.anchor = 5L, verbose = FALSE)

transferred <- TransferData(
  anchorset = anchors,
  refdata = GetAssayData(reference, assay = "ADT", layer = "data"),
  weight.reduction = "pcaproject", dims = seq_len(n_dims), verbose = FALSE)

prediction <- t(as.matrix(GetAssayData(transferred, layer = "data")))
stopifnot(nrow(prediction) == nrow(query_rna),
          ncol(prediction) == n_proteins)
prediction[!is.finite(prediction)] <- 0

writeBin(as.numeric(t(prediction)), out_path, size = 4)
cat(sprintf("Seurat wrote %d x %d\n", nrow(prediction), ncol(prediction)))
