# ICPR_2026
Code and experiments for the ICPR 2026 submission on frugal vs deep learning approaches for text-based prediction under domain shift and ordinal constraints.

## Abstract
Technology Readiness Levels (TRLs) are an ordinal maturity
scale widely used in innovation assessment, yet infering them from text is
hindered by scarce labeled data and domain-dependent conventions. We
formulate TRL inference as a latent ordinal pattern recognition problem
under weak supervision. We introduce a large silver corpus constructed
from project descriptions using hybrid symbolic–semantic labeling, and
reconstruction rules to improve alignment with expert annotations. We
compare flat classification, regression, ordinal decomposition, and hierarchical
models using ordinal error metrics. Linear models recover a coherent
ordinal pattern in-domain, with errors concentrated within ±1
level. We show that this structure degrades under zero-shot transfer to
an expert gold corpus, revealing a strong semantic domain shift. We
also show that rigid ordinal constraints are brittle under transfer, while
soft hierarchical routing and lightweight hybrid ensembling significantly
reduce large ordinal errors. Analyses of grammatical and structural representations
indicate that domain shift is driven primarily by semantic
conventions rather than syntax. These results highlight both the potential
and the limits of weakly supervised ordinal pattern recognition in
real-world settings. To foster reproducibility, we release the dataset and
the code in this Github repository.

## Objectives
- Investigate frugal models (linear SVM / logistic regression) with lexical and structural features.
- Compare classification vs regression formulations, including ordinal-specific strategies.
- Analyse the contribution of grammatical and structural features vs full lexical features.
- Study few-shot learning and domain shift (silver vs gold labels, cross-domain evaluation).
- Evaluate deep learning approaches based on BERT and SciBERT, including zero/few-shot regimes.
- Provide visual analyses of model behaviour (n-grams, bias, domain shift visualisation).

## Repository Structure

- `requirements.txt` 
  Provide requirements, Python version : 3.12.10

- `01_data/`  
  Scripts and utilities related to data preparation.  
  - `01b_build_silver_minimize.py` (expected): builds a constrained “silver” version of the dataset from raw resources.

- `02_baseline/`  
  Frugal baseline classifiers with lexical features.  
  - `20_classification_full.py`: full-feature baseline classification.  
  - `20b_svm_logreg_sweep.py`: hyperparameter sweeps for SVM and logistic regression.  
  - `20d_svm_logreg_char_ngrams.py`: character n‑gram based SVM / logistic regression.

- `03_ordinality/`  
  Methods that explicitly model the ordinal structure of the labels.  
  - `20c_two_stage_svm.py`: two-stage SVM approach for ordinal prediction.  
  - `20e_stacking_svm_rf.py`: stacked SVM + random forest models.  
  - `20f_ordinal_decomposition.py`: ordinal decomposition strategies.

- `04_grammar/`  
  Grammatical and structural feature models.  
  - `21_classification_grammar_trl.py`: grammar-based classification.  
  - `22_classification_fusion_full_grammar.py`: fusion of full and grammar-based features.  
  - `25_structural_features_grammar.py`: structural and grammatical feature extraction.  
  - `26_compare_full_vs_grammar_disagreements.py`: analysis of disagreements between full vs grammar-based models.

  - `05_regression/`  
  Regression-based formulations of the prediction problem.  
  - `30_regression_full.py`: full-feature regression setup.  
  - `31_regression_fewshot.py`: few-shot regression experiments.

- `06_domainshift/`  
  Few-shot and domain-shift analyses.  
  - `21_classification_fewshot.py`: few-shot classification experiments.  
  - `23_disagreement_silver_vs_iea.py`: disagreement analysis between silver labels and reference annotations.  
  - `50_train_svm_on_gold.py`: training SVM baselines on gold-labeled data.

- `07_deeplearning/`  
  Transformer-based models and deep learning experiments.  
  - `40_bert_vs_scibert.py`: comparison between BERT and SciBERT.  
  - `41_scibert_master.py`: main SciBERT training and evaluation pipeline.  
  - `42_scibert_fewshot.py`: few-shot experiments with SciBERT.


## Datasets

The final datasets used in the experiments are stored in the `dataset/` folder:

- `dataset/IEA_Clean_Guide.csv`  
  Expert-annotated **Gold** corpus from IEA Clean Energy Technology Guide. (as available on their website)

- `dataset/minimized_silver_dataset.csv`  
  Automatically labeled **Silver** corpus built from CORDIS project summaries using the hybrid weak supervision pipeline.

- `dataset/IEA_Clean_Guide_Final_with_Text.csv`  
  Expert-annotated **Gold** corpus derived from the IEA Clean Energy Technology Guide, used as the target evaluation domain.

All experiment scripts resolve these files relative to the repository root, so the commands below should be run from the root directory after cloning the repository.

# Reproducibility

This repository contains all scripts used to produce the experiments reported in the paper.

## Requirements

- Python 3.12.10
- Install all dependencies with

```bash
pip install -r requirements.txt
```

## Quick Start

Clone the repository

```bash
git clone ...

cd ICPR_2026
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Running the experiments

The experiments are independent and can be executed separately.

## 1. Frugal lexical baselines

Run the main linear SVM / Logistic Regression experiments:

```bash
python 02_baseline/20_classification_full.py
```

Hyperparameter search:

```bash
python 02_baseline/20b_svm_logreg_sweep.py
```

Character n-gram experiments:

```bash
python 02_baseline/20d_svm_logreg_char_ngrams.py
```

---

## 2. Ordinal learning experiments

Two-stage ordinal SVM:

```bash
python 03_ordinality/20c_two_stage_svm.py
```

Stacked SVM + Random Forest:

```bash
python 03_ordinality/20e_stacking_svm_rf.py
```

Ordinal decomposition:

```bash
python 03_ordinality/20f_ordinal_decomposition.py
```

---

## 3. Grammar and structural representations

Grammar-based representation:

```bash
python 04_grammar/21_classification_grammar_trl.py
```

Fusion representation:

```bash
python 04_grammar/22_classification_fusion_full_grammar.py
```

Structural features:

```bash
python 04_grammar/25_structural_features_grammar.py
```

Grammar/full-text disagreement analysis:

```bash
python 04_grammar/26_compare_full_vs_grammar_disagreements.py
```

---

## 4. Regression models

Full regression:

```bash
python 05_regression/30_regression_full.py
```

Few-shot regression:

```bash
python 05_regression/31_regression_fewshot.py
```

---

## 5. Domain-shift experiments

Few-shot transfer:

```bash
python 06_domainshift/21_classification_fewshot.py
```

Silver vs Gold disagreement analysis:

```bash
python 06_domainshift/23_disagreement_silver_vs_iea.py
```

Gold-domain SVM:

```bash
python 06_domainshift/50_train_svm_on_gold.py
```

---

## 6. Deep learning experiments

BERT vs SciBERT:

```bash
python 07_deeplearning/40_bert_vs_scibert.py
```

Main SciBERT training:

```bash
python 07_deeplearning/41_scibert_master.py
```

SciBERT few-shot adaptation:

```bash
python 07_deeplearning/42_scibert_fewshot.py
```

## Results

The repository includes generated result folders for the classical, ordinal, grammar, regression, domain-shift, and deep-learning experiments. These folders contain the CSV metric tables, saved checkpoints, and figures produced by the scripts. The deep-learning scripts require Hugging Face model downloads and substantially longer runtimes.

### Result folders

| Folder | Produced by | Main contents |
|--------|-------------|---------------|
| `20_results_dual_eval/` | `python 02_baseline/20_classification_full.py` | `dual_metrics.csv` and confusion matrices for Silver and IEA evaluation. |
| `20b_sweep_results/` | `python 02_baseline/20b_svm_logreg_sweep.py` | `sweep_svm_logreg.csv`, Silver/Gold confusion matrices, and `scatter_silver_vs_gold_relaxed.png`. |
| `20d_char_ngrams_results/` | `python 02_baseline/20d_svm_logreg_char_ngrams.py` | `char_vs_word_results.csv`. |
| `20c_two_stage_results/` | `python 03_ordinality/20c_two_stage_svm.py` | `metrics_two_stage_vs_baseline.csv`. |
| `20e_stacking_results/` | `python 03_ordinality/20e_stacking_svm_rf.py` | `metrics_stacking_svm_rf.csv`. |
| `20f_ordinal_decomposition_results/` | `python 03_ordinality/20f_ordinal_decomposition.py` | `metrics_ordinal_decomposition.csv`. |
| `21_results_dual_eval_grammar/` | `python 04_grammar/21_classification_grammar_trl.py` | `dual_metrics_grammar.csv` and grammar-model confusion matrices. |
| `22_results_dual_eval_fusion/` | `python 04_grammar/22_classification_fusion_full_grammar.py` | `dual_metrics_fusion.csv` and fusion-model confusion matrices. |
| `25_results_structural_grammar/` | `python 04_grammar/25_structural_features_grammar.py` | `metrics_structural_grammar.csv`, Gold confusion matrices, and SVM R1/R2 IEA prediction files. |
| `26_results_full_vs_grammar/` | `python 04_grammar/26_compare_full_vs_grammar_disagreements.py` | `pattern_error_stats_R1_vs_R2.csv` and bias/error-distribution figures. |
| `30_results_regression_comparison/` | `python 05_regression/30_regression_full.py` | `metrics_regression_comparison.csv` and regression/classification confusion matrices. |
| `31_results_regression_fewshot/` | `python 05_regression/31_regression_fewshot.py` | `metrics_regression_fewshot.csv`. |
| `21_results_fewshot_all/` | `python 06_domainshift/21_classification_fewshot.py` | `metrics_fewshot.csv`. |
| `23_results_disagreement/` | `python 06_domainshift/23_disagreement_silver_vs_iea.py` | IEA disagreement examples and label-level disagreement summary. |
| `50_results_svm_iea_to_silver/` | `python 06_domainshift/50_train_svm_on_gold.py` | `metrics_svm_iea_to_silver.csv` and IEA-to-Silver confusion matrix. |
| `40_results_bert_vs_scibert/` | `python 07_deeplearning/40_bert_vs_scibert.py` | `metrics_comparison.csv` and BERT/SciBERT confusion matrices on IEA. |
| `41_scibert_final_model/` | `python 07_deeplearning/41_scibert_master.py` | Saved SciBERT checkpoint and `final/` model used by the few-shot script. |
| `scibert_fewshot_output/` | `python 07_deeplearning/42_scibert_fewshot.py` | Best/checkpoint output from SciBERT few-shot adaptation. |
| `43_deep_all_results/` | `python 07_deeplearning/43_deep_bert_scibert_full_fewshot.py` | `metrics_deep_all_scenarios.csv`, comparing BERT/SciBERT zero-shot and few-shot settings. |

### Deep learning execution

The four scripts in `07_deeplearning/` were run on a local NVIDIA RTX A2000 8GB Laptop GPU. The environment used a CUDA-enabled PyTorch build (`torch 2.11.0+cu128`) and Hugging Face `accelerate`; CPU execution is possible but much slower.

| Script | Purpose | Observed output |
|--------|---------|-----------------|
| `40_bert_vs_scibert.py` | Trains BERT base and SciBERT on the full Silver corpus, then evaluates zero-shot transfer on IEA. | Writes `40_results_bert_vs_scibert/metrics_comparison.csv`. BERT: strict `15.7%`, relaxed `59.6%`, MAE `1.84`, train time `5271s`. SciBERT: strict `14.9%`, relaxed `54.3%`, MAE `2.08`, train time `5292s`. |
| `41_scibert_master.py` | Trains the main SciBERT model on a balanced Silver subset, evaluates it on IEA, saves the model for later few-shot adaptation, and generates diagnostic figures. | Saves `41_scibert_final_model/final/`. IEA evaluation before the figure step: strict `17.2%`, relaxed `40.9%`, macro-F1 `0.110`. It generated `figure1_learning_curve.png`, `figure2_accuracy_evolution.png`, and `figure3_confusion_matrix.png`; the final t-SNE step failed in the provided run because tokenized tensors were left on CPU while the model was on CUDA. The model checkpoint was saved before that error. |
| `42_scibert_fewshot.py` | Loads `41_scibert_final_model/final`, fine-tunes on 50 IEA examples, and evaluates on the remaining 554 IEA examples. | Writes checkpoints under `scibert_fewshot_output/`. Final evaluation: strict `34.8%`, relaxed `56.0%`, macro-F1 `0.162`, eval loss `2.009`. |
| `43_deep_bert_scibert_full_fewshot.py` | Runs the most complete deep-learning comparison: BERT and SciBERT, zero-shot Silver-to-IEA, then few-shot adaptation on 50 IEA examples. | Writes `43_deep_all_results/metrics_deep_all_scenarios.csv`. On Gold IEA, BERT zero-shot reaches relaxed `55.1%`, BERT few-shot `57.6%`, SciBERT zero-shot `54.5%`, and SciBERT few-shot `51.3%`. |

The most complete deep-learning result file is `43_deep_all_results/metrics_deep_all_scenarios.csv`:

| Model | Scenario | Dataset | Strict | Relaxed | MAE | Cat>=3 | F1 macro |
|-------|----------|---------|--------|---------|-----|--------|----------|
| BERT base | Zero-shot | Silver test | `0.621` | `0.705` | `1.34` | `0.251` | `0.432` |
| BERT base | Zero-shot | Gold IEA test | `0.153` | `0.551` | `1.95` | `0.332` | `0.096` |
| BERT base | Few-shot | Silver test | `0.430` | `0.649` | `1.75` | `0.302` | `0.353` |
| BERT base | Few-shot | Gold IEA test | `0.370` | `0.576` | `1.64` | `0.283` | `0.208` |
| SciBERT | Zero-shot | Silver test | `0.660` | `0.736` | `1.20` | `0.222` | `0.507` |
| SciBERT | Zero-shot | Gold IEA test | `0.143` | `0.545` | `2.04` | `0.352` | `0.092` |
| SciBERT | Few-shot | Silver test | `0.467` | `0.678` | `1.61` | `0.274` | `0.409` |
| SciBERT | Few-shot | Gold IEA test | `0.309` | `0.513` | `1.93` | `0.347` | `0.169` |

### Main reviewer checks

The following commands are the main reviewer-oriented entry points. Each script prints a summary table and writes CSV metrics and/or figures in the corresponding result folder.

| Paper result to check | Command | Main output to compare |
|--------------|---------|------------------------|
| Frugal Silver-to-IEA baselines discussed in the main domain-shift results | `python 02_baseline/20_classification_full.py` | Console metrics and `20_results_dual_eval/dual_metrics.csv`: compare strict accuracy, relaxed accuracy (`+/-1`), MAE, and catastrophic errors on Silver and Gold/IEA. |
| Ordinal modelling and hierarchical SVM results | `python 03_ordinality/20c_two_stage_svm.py` | `20c_two_stage_results/metrics_two_stage_vs_baseline.csv`: compare baseline one-shot SVM with the two-stage ordinal SVM. |
| Regression formulation of TRL prediction | `python 05_regression/30_regression_full.py` | `30_results_regression_comparison/metrics_regression_comparison.csv`: compare regression MAE and relaxed accuracy with the classification baselines. |
| Few-shot target-domain adaptation with classical models | `python 06_domainshift/21_classification_fewshot.py` | `21_results_fewshot_all/metrics_fewshot.csv`: compare the remaining IEA test set after adding 50 boosted IEA examples to the Silver training set. |
| Structural/grammar representations and grammar-vs-full-text comparison | `python 04_grammar/25_structural_features_grammar.py` then `python 04_grammar/26_compare_full_vs_grammar_disagreements.py` | `25_results_structural_grammar/metrics_structural_grammar.csv`, `25_results_structural_grammar/svm_R1_fulltext_preds_iea.csv`, `25_results_structural_grammar/svm_R2_grammar_preds_iea.csv`, and `26_results_full_vs_grammar/pattern_error_stats_R1_vs_R2.csv`: compare R1 full text, R2 grammar+TRL, R3 structural features, and R4 full+structural results with the grammar analysis and bias-pattern figure. |
| BERT/SciBERT zero-shot Silver-to-IEA transfer | `python 07_deeplearning/40_bert_vs_scibert.py` | `40_results_bert_vs_scibert/metrics_comparison.csv`: compare BERT and SciBERT zero-shot metrics with the deep-learning table. |
| Main SciBERT run used for zero-shot SciBERT analysis | `python 07_deeplearning/41_scibert_master.py` | `41_scibert_final_model/final/` and generated learning-curve, accuracy, confusion-matrix, and t-SNE figures: compare SciBERT Silver validation and IEA evaluation with the SciBERT rows of the deep-learning table. |
| SciBERT few-shot adaptation on 50 IEA examples | `python 07_deeplearning/42_scibert_fewshot.py` | Console evaluation metrics after few-shot fine-tuning: compare the IEA few-shot SciBERT row of the deep-learning table. |

### Summary of reproduced classical results

The terminal outputs used to generate the included result folders show the following key values. Results may vary slightly across Python/scikit-learn versions and hardware, but the qualitative conclusions should remain the same.

| Experiment | Best or reference outcome on IEA/Gold |
|------------|----------------------------------------|
| Full lexical baselines (`20_classification_full.py`) | Linear SVM: strict `12.9%`, relaxed `50.3%`; Random Forest: strict `12.4%`, relaxed `66.6%`. |
| SVM/LogReg sweep (`20b_svm_logreg_sweep.py`) | Best Gold relaxed configuration: SVM `C=2.0`, ngram `(1, 3)`, `max_features=50000`, relaxed `52.2%`. |
| Word vs character n-grams (`20d_svm_logreg_char_ngrams.py`) | Word SVM remains strongest among tested variants: Gold strict `12.9%`, relaxed `50.3%`, MAE `2.09`. |
| Two-stage ordinal SVM (`20c_two_stage_svm.py`) | Soft two-stage improves Gold relaxed accuracy from `50.3%` to `65.4%` and lowers MAE from `2.09` to `1.72`. |
| SVM/RF stacking (`20e_stacking_svm_rf.py`) | Best Gold mixture uses SVM alpha `0.40`, MAE `1.63`, catastrophic error `23.7%`. |
| Ordinal decomposition (`20f_ordinal_decomposition.py`) | Gold strict `11.6%`, relaxed `53.8%`, MAE `1.87`, catastrophic error `29.8%`. |
| Grammar + TRL terms (`21_classification_grammar_trl.py`) | SVM improves Silver relaxed accuracy (`67.1%`) but drops to Gold relaxed `43.7%`; RF reaches Gold relaxed `65.1%`. |
| Full + grammar fusion (`22_classification_fusion_full_grammar.py`) | Fusion SVM reaches Silver relaxed `69.9%` but only Gold relaxed `44.2%`; RF reaches Gold relaxed `66.4%`. |
| Structural grammar representations (`25_structural_features_grammar.py`) | Full-text SVM R1: Gold MAE `2.09`, relaxed `50.3%`; grammar R2: Gold MAE `2.54`, relaxed `43.9%`; RF variants reach Gold relaxed around `65.7%`-`66.6%`. |
| Full-vs-grammar disagreement analysis (`26_compare_full_vs_grammar_disagreements.py`) | R1 full-text: MAE `2.09`, relaxed `50.3%`, catastrophic `37.4%`; R2 grammar: MAE `2.54`, relaxed `43.9%`, catastrophic `43.0%`. |
| Regression comparison (`30_regression_full.py`) | Linear SVR gives Gold MAE `1.85` and relaxed `37.6%`; Logistic Regression gives Gold relaxed `41.6%` but MAE `2.36`. |
| Regression few-shot (`31_regression_fewshot.py`) | Linear SVR few-shot: Gold hidden relaxed `45.1%`, MAE `1.74`. |
| Classification few-shot (`21_classification_fewshot.py`) | Random Forest on Silver plus 50 boosted IEA examples: hidden IEA relaxed `66.8%`, MAE `1.63`. |
| Silver-vs-IEA disagreement (`23_disagreement_silver_vs_iea.py`) | Silver-trained SVM accuracy on IEA hidden test: `11.7%`; IEA few-shot SVM accuracy: `42.2%`; model disagreements: `96.4%`. |
| IEA-to-Silver reverse transfer (`50_train_svm_on_gold.py`) | SVM trained on IEA and tested on Silver: strict `3.3%`, relaxed `47.5%`. |

Notes for reviewers:

- `04_grammar/26_compare_full_vs_grammar_disagreements.py` depends on the prediction files generated by `04_grammar/25_structural_features_grammar.py`; run `25` first.
- The deep-learning scripts download Hugging Face models and are substantially slower than the classical models. The paper reports these experiments on a single NVIDIA T4 GPU; CPU execution is possible but slow.
- The most important metrics for comparison with the paper are relaxed accuracy (`+/-1` TRL), MAE, and catastrophic error rate. Strict accuracy is reported, but the paper emphasizes ordinal robustness rather than exact-match accuracy alone.
