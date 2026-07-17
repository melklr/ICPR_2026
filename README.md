# ICPR_2026

Code, datasets, and reproducible experiments accompanying the ICPR 2026 paper on Technology Readiness Level (TRL) prediction from text under weak supervision, domain shift, and ordinal constraints.

## Project Description

Technology Readiness Levels (TRLs) are an ordinal measure of technology maturity. 
This repository studies how well TRL can be inferred from project or technology descriptions when labeled data are scarce
and domain conventions differ.

This repository reproduces experiments on:
- frugal lexical models, including linear SVM and logistic regression;
- regression and ordinal formulations of the TRL detection task;
- grammatical, structural, and feature representations;
- silver-to-gold domain shift between CORDIS-derived data and IEA annotations;
- BERT and SciBERT zero-shot and few-shot transfer.

The main evaluation metrics are strict accuracy, relaxed accuracy within
`+/-1` TRL level, mean absolute error (MAE), catastrophic error rate
(`>=3` TRL levels), and macro-F1.

Minor numerical differences may still occur across hardware and library versions.

## Installation

The experiments were run with Python `3.12.10`.

```bash
git clone https://github.com/melklr/ICPR_2026.git
cd ICPR_2026
pip install -r requirements.txt
```


## Dataset

The final datasets used by the scripts are stored in `dataset/`:

| Dataset | Description |
|---------|-------------|
| `minimized_silver_dataset.csv` | Silver corpus used for classical experiments |
| `silver_dataset_master.csv` | Silver corpus used for deep learning |
| `IEA_Clean_Guide_Final_with_Text.csv` | Gold evaluation corpus |


## Repository Structure

| Folder | Description |
|--------|-------------|
| `01_data` | Data preparation |
| `02_baseline` | Classical baselines |
| `03_ordinality` | Ordinal models |
| `04_grammar` | Grammar and structural features |
| `05_regression` | Regression models |
| `06_domainshift` | Domain-shift and few-shot experiments |
| `07_deeplearning` | BERT and SciBERT experiments |
| `all_results_output` | Generated metrics, figures and checkpoints |

## Reproduce Paper Results

Install the requirements first, then run the desired scripts from the repository root. The classical scripts are independent unless otherwise noted.

Reference outputs reported in the paper are provided in `all_results_output/` for convenience. Running the scripts reproduces the experiments and writes newly generated outputs to the repository root (`ICPR_2026/`), preserving the reference results for comparison.


| Paper result | Script |
|--------------|--------|
| Weakly supervised lexical baselines | `02_baseline/20_classification_full.py` |
| Hyperparameter optimization | `02_baseline/20b_svm_logreg_sweep.py` |
| Word vs. character n-grams | `02_baseline/20d_svm_logreg_char_ngrams.py` |
| Two-stage ordinal SVM | `03_ordinality/20c_two_stage_svm.py` |
| SVM / Random Forest stacking | `03_ordinality/20e_stacking_svm_rf.py` |
| Ordinal decomposition | `03_ordinality/20f_ordinal_decomposition.py` |
| Grammar-aware representation | `04_grammar/21_classification_grammar_trl.py` |
| Grammar + lexical fusion | `04_grammar/22_classification_fusion_full_grammar.py` |
| Structural grammatical features | `04_grammar/25_structural_features_grammar.py` |
| Linguistic error analysis (Fig. 3) | `04_grammar/26_compare_full_vs_grammar_disagreements.py` |
| Regression experiments | `05_regression/30_regression_full.py` |
| Few-shot regression | `05_regression/31_regression_fewshot.py` |
| Classical few-shot adaptation | `06_domainshift/21_classification_fewshot.py` |
| Silver vs. Gold disagreement analysis | `06_domainshift/23_disagreement_silver_vs_iea.py` |
| Gold → Silver transfer | `06_domainshift/50_train_svm_on_gold.py` |
| BERT vs. SciBERT comparison | `07_deeplearning/40_bert_vs_scibert.py` |
| Final SciBERT model | `07_deeplearning/41_scibert_master.py` |
| SciBERT few-shot adaptation | `07_deeplearning/42_scibert_fewshot.py` |
| Complete deep-learning benchmark (Table 1) | `07_deeplearning/43_deep_bert_scibert_full_fewshot.py` |

## Deep Learning

Deep-learning scripts also require a working PyTorch installation. GPU execution
is recommended. The reproduced deep-learning outputs were generated on an
NVIDIA RTX A2000 8GB Laptop GPU with a CUDA-enabled PyTorch build and
Hugging Face `accelerate`.

To check whether PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```


| Script | Purpose | 
|--------|---------|
| `07_deeplearning/40_bert_vs_scibert.py` | Train BERT base and SciBERT on the full Silver corpus and evaluate zero-shot transfer on IEA. | 
| `07_deeplearning/41_scibert_master.py` | Train the main SciBERT model on Silver, evaluate on IEA, save the model, and generate diagnostic figures. | 
| `07_deeplearning/42_scibert_fewshot.py` | Load the saved SciBERT model, fine-tune on 50 IEA examples, and evaluate on the remaining IEA examples. | 
| `07_deeplearning/43_deep_bert_scibert_full_fewshot.py` | Run the complete BERT/SciBERT zero-shot and few-shot comparison. | 

Hardware

- NVIDIA RTX A2000 Laptop GPU (8 GB)
- CUDA-enabled PyTorch

Typical runtime

- ~90 min/model

Outputs

- trained checkpoints
- metrics CSV
- figures

Observed deep-learning results:

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



## Citation

If you use this repository, please cite the associated ICPR 2026 paper. The
final citation will be added after publication.

```bibtex
@inproceedings{icpr2026_trl,
  title = {Learning and Recognizing Latent Innovation Maturity Indicator Patterns in Texts},
  author = {Mélusine Caillard, Gaël Lejeune, Pierre-Emmanuel Fayemi, and Aoussat Améziane},
  booktitle = {Proceedings of ICPR},
  year = {2026}
}
```
