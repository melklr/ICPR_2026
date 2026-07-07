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

## Expected results

The following scripts reproduce the main results reported in the paper.

| Paper result | Script |
|--------------|--------|
| Classical baselines | 02_baseline/20_classification_full.py |
| Hyperparameter search | 02_baseline/20b_svm_logreg_sweep.py |
| Ordinal models | 03_ordinality/20c_two_stage_svm.py |
| Grammar experiments | 04_grammar/21_classification_grammar_trl.py |
| Regression experiments | 05_regression/30_regression_full.py |
| Few-shot adaptation | 06_domainshift/21_classification_fewshot.py |
| BERT / SciBERT | 07_deeplearning/41_scibert_master.py |
