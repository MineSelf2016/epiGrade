PY       := .venv/bin/python
FIGSHARE := https://figshare.com/ndownloader/articles/34332525/versions/1
SETTINGS := cbmc:CBMC:cell eccite_5p:ECCITE:cell eccite_5p:ECCITE:donor \
            gse3p_3p:GSE3P:cell gse3p_3p:GSE3P:donor

.PHONY: data figshare legacy cluster baselines analysis test

data:
	bash code/download_data.sh
	bash code/download_external.sh
	$(PY) code/experiments/prepare_data.py --dataset all

figshare:
	curl -fL -o epigrade_figshare.zip "$(FIGSHARE)"
	unzip -o epigrade_figshare.zip
	@for f in epigrade_*.tar.gz; do tar -xzf "$$f"; done

legacy:
	@set -e; for s in $(SETTINGS); do \
	  f=$${s%%:*}; r=$${s#*:}; n=$${r%%:*}; m=$${r#*:}; \
	  lodo=""; [ "$$m" = donor ] && lodo="--lodo"; \
	  $(PY) code/experiments/run_epigrade.py --data data/processed/$$f.h5ad \
	    --dataset-name $$n --split-mode $$m $$lodo --variant full \
	    --output results_epigrade/$${n}_$${m}_full; \
	done
	@set -e; for s in eccite_5p:ECCITE gse3p_3p:GSE3P; do \
	  f=$${s%%:*}; n=$${s#*:}; \
	  $(PY) code/experiments/run_epigrade.py --data data/processed/$$f.h5ad \
	    --dataset-name $$n --split-mode donor --variant full \
	    --output results_epigrade/donor5/$${n}_donor_full; \
	done

cluster:
	mkdir -p logs results_fused/.locks
	sbatch --array=1-$$(wc -l < jobs/m_fused.txt) jobs/run_fused.sbatch
	sbatch --array=1-3 jobs/run_xc_fused.sbatch
	sbatch --array=1-3 jobs/run_xc_consensus_fused.sbatch
	@echo "when run_fused has finished: sbatch --array=1-4 jobs/run_annot_fused.sbatch"

baselines:
	@set -e; for s in $(SETTINGS); do \
	  f=$${s%%:*}; r=$${s#*:}; n=$${r%%:*}; m=$${r#*:}; \
	  $(PY) code/experiments/run_baselines.py --data data/processed/$$f.h5ad \
	    --dataset-name $$n --reference-run results_epigrade/$${n}_$${m}_full \
	    --output results_epigrade/$${n}_$${m}_external \
	    --external scipenn totalvi seurat_r; \
	  $(PY) code/experiments/run_baselines.py --data data/processed/$$f.h5ad \
	    --dataset-name $$n --reference-run results_fused/$${n}_$${m}_full \
	    --output results_fused/knn30/$${n}_$${m} --config code/configs/fused.yaml \
	    --only knn_expression knn_pca; \
	done
	bash jobs/knn30_donor5.sh

analysis:
	$(PY) code/experiments/stage_fused_results.py
	$(PY) code/experiments/aggregate_folds.py
	$(PY) code/experiments/analyze_epigrade.py
	$(PY) code/experiments/prior_leakage.py
	$(PY) code/experiments/protein_reliability.py
	$(PY) code/experiments/grade_transfer.py
	$(PY) code/experiments/make_supplementary.py
	$(PY) code/experiments/make_supp_extra.py
	bash jobs/figures_fused.sh

test:
	$(PY) -m pytest code/tests -q
	$(PY) code/scripts/audit_release.py
