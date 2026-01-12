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

- `dataset/minimized_silver_dataset.csv`  
  Automatically labeled **Silver** corpus built from CORDIS project summaries using the hybrid weak supervision pipeline.

- `dataset/IEA_Clean_Guide_Final_with_Text.csv`  
  Expert-annotated **Gold** corpus derived from the IEA Clean Energy Technology Guide, used as the target evaluation domain.
