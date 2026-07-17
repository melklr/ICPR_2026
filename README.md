# ICPR_2026

Code, datasets, and experiment outputs for the ICPR 2026 submission on
text-based Technology Readiness Level (TRL) prediction under weak supervision,
domain shift, and ordinal constraints.

## Project Description

Technology Readiness Levels (TRLs) are ordinal maturity labels used in
innovation assessment. This repository studies how well TRL values can be
predicted from project or technology descriptions when labeled data are scarce
and domain conventions differ.

The paper compares:

- frugal lexical models, including linear SVM and logistic regression;
- regression and ordinal formulations of the TRL task;
- grammatical, structural, and fused feature representations;
- silver-to-gold domain shift between CORDIS-derived data and IEA annotations;
- BERT and SciBERT zero-shot and few-shot transfer.

The main evaluation metrics are strict accuracy, relaxed accuracy within
`+/-1` TRL level, mean absolute error (MAE), catastrophic error rate
(`>=3` TRL levels), and macro-F1.

## Installation

The experiments were run with Python `3.12.10`.

```bash
git clone https://github.com/melklr/ICPR_2026.git
cd ICPR_2026
pip install -r requirements.txt
```

Deep-learning scripts also require a working PyTorch installation. GPU execution
is recommended. The reproduced deep-learning outputs were generated on an
NVIDIA RTX A2000 8GB Laptop GPU with a CUDA-enabled PyTorch build and
Hugging Face `accelerate`.

To check whether PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset

The final datasets used by the scripts are stored in `dataset/`:

| File | Role |
|------|------|
| `dataset/minimized_silver_dataset.csv` | Silver corpus built from CORDIS project summaries using the weak-supervision pipeline. |
| `dataset/silver_dataset_master.csv` | Larger silver dataset used by the deep-learning scripts. |
| `dataset/IEA_Clean_Tech_Guide.csv` | Raw IEA Clean Energy Technology Guide file. |
| `dataset/IEA_Clean_Guide_Final_with_Text.csv` | Expert-annotated IEA gold corpus used as the target evaluation domain. |

All scripts resolve dataset paths relative to the repository root. Run commands
from the repository root unless stated otherwise.

## Repository Structure

| Path | Contents |
|------|----------|
| `01_data/` | Dataset preparation and cleaning scripts. |
| `02_baseline/` | Lexical SVM, logistic regression, random forest, and n-gram baselines. |
| `03_ordinality/` | Ordinal learning variants: two-stage SVM, stacking, and ordinal decomposition. |
| `04_grammar/` | Grammar, structural, fusion, and full-vs-grammar disagreement analyses. |
| `05_regression/` | Regression formulations and few-shot regression. |
| `06_domainshift/` | Few-shot transfer, silver-vs-IEA disagreement, and gold-to-silver transfer. |
| `07_deeplearning/` | BERT/SciBERT zero-shot and few-shot experiments. |
| `all_results_output/` | Generated CSV metrics, figures, and saved deep-learning checkpoints. |

## Reproduce Paper Results

Install the requirements first, then run the scripts below from the repository
root. The classical scripts are independent unless noted. Each script prints a
summary and writes outputs under `all_results_output/`.

### Main reviewer entry points

| Paper result to check | Command | Output to compare |
|-----------------------|---------|-------------------|
| Frugal lexical baselines and silver-to-IEA domain shift | `python 02_baseline/20_classification_full.py` | `all_results_output/20_results_dual_eval/dual_metrics.csv` |
| SVM/logistic regression hyperparameter sweep | `python 02_baseline/20b_svm_logreg_sweep.py` | `all_results_output/20b_sweep_results/sweep_svm_logreg.csv` |
| Character n-gram comparison | `python 02_baseline/20d_svm_logreg_char_ngrams.py` | `all_results_output/20d_char_ngrams_results/char_vs_word_results.csv` |
| Two-stage ordinal SVM | `python 03_ordinality/20c_two_stage_svm.py` | `all_results_output/20c_two_stage_results/metrics_two_stage_vs_baseline.csv` |
| SVM/RF stacking | `python 03_ordinality/20e_stacking_svm_rf.py` | `all_results_output/20e_stacking_results/metrics_stacking_svm_rf.csv` |
| Ordinal decomposition | `python 03_ordinality/20f_ordinal_decomposition.py` | `all_results_output/20f_ordinal_decomposition_results/metrics_ordinal_decomposition.csv` |
| Grammar-only representation | `python 04_grammar/21_classification_grammar_trl.py` | `all_results_output/21_results_dual_eval_grammar/dual_metrics_grammar.csv` |
| Full-text plus grammar fusion | `python 04_grammar/22_classification_fusion_full_grammar.py` | `all_results_output/22_results_dual_eval_fusion/dual_metrics_fusion.csv` |
| Structural grammar analysis | `python 04_grammar/25_structural_features_grammar.py` | `all_results_output/25_results_structural_grammar/metrics_structural_grammar.csv` |
| Full-vs-grammar disagreement analysis | `python 04_grammar/26_compare_full_vs_grammar_disagreements.py` | `all_results_output/26_results_full_vs_grammar/pattern_error_stats_R1_vs_R2.csv` |
| Regression formulation | `python 05_regression/30_regression_full.py` | `all_results_output/30_results_regression_comparison/metrics_regression_comparison.csv` |
| Few-shot regression | `python 05_regression/31_regression_fewshot.py` | `all_results_output/31_results_regression_fewshot/metrics_regression_fewshot.csv` |
| Classical few-shot target adaptation | `python 06_domainshift/21_classification_fewshot.py` | `all_results_output/21_results_fewshot_all/metrics_fewshot.csv` |
| Silver-vs-IEA disagreement analysis | `python 06_domainshift/23_disagreement_silver_vs_iea.py` | `all_results_output/23_results_disagreement/` |
| Gold-to-silver reverse transfer | `python 06_domainshift/50_train_svm_on_gold.py` | `all_results_output/50_results_svm_iea_to_silver/metrics_svm_iea_to_silver.csv` |

Run `04_grammar/25_structural_features_grammar.py` before
`04_grammar/26_compare_full_vs_grammar_disagreements.py`, because script `26`
uses the prediction files generated by script `25`.

### Summary of reproduced classical results

| Experiment | Reference IEA/Gold outcome |
|------------|----------------------------|
| Full lexical baselines | Linear SVM: strict `12.9%`, relaxed `50.3%`; Random Forest: relaxed `66.6%`. |
| SVM/logistic regression sweep | Best Gold relaxed result: SVM `C=2.0`, ngram `(1, 3)`, `max_features=50000`, relaxed `52.2%`. |
| Word vs character n-grams | Word SVM: strict `12.9%`, relaxed `50.3%`, MAE `2.09`. |
| Two-stage ordinal SVM | Soft two-stage improves relaxed accuracy from `50.3%` to `65.4%` and MAE from `2.09` to `1.72`. |
| SVM/RF stacking | Best Gold mixture: SVM alpha `0.40`, MAE `1.63`, catastrophic error `23.7%`. |
| Ordinal decomposition | Strict `11.6%`, relaxed `53.8%`, MAE `1.87`, catastrophic error `29.8%`. |
| Grammar + TRL terms | SVM reaches Gold relaxed `43.7%`; RF reaches Gold relaxed `65.1%`. |
| Full + grammar fusion | Fusion SVM reaches Gold relaxed `44.2%`; RF reaches Gold relaxed `66.4%`. |
| Structural grammar representations | Full-text SVM R1: relaxed `50.3%`, MAE `2.09`; grammar R2: relaxed `43.9%`, MAE `2.54`. |
| Regression comparison | Linear SVR: Gold MAE `1.85`, relaxed `37.6%`. |
| Few-shot classification | Random Forest with 50 IEA examples: hidden IEA relaxed `66.8%`, MAE `1.63`. |
| IEA-to-silver reverse transfer | SVM trained on IEA and tested on Silver: strict `3.3%`, relaxed `47.5%`. |

Results may vary slightly across Python, scikit-learn, PyTorch, and hardware
versions. The paper focuses mainly on relaxed accuracy, MAE, and catastrophic
error rate rather than strict accuracy alone.

## Deep Learning

The deep-learning scripts download Hugging Face models and run much longer than
the classical experiments. CPU execution is possible but slow.

| Script | Purpose | Main output |
|--------|---------|-------------|
| `07_deeplearning/40_bert_vs_scibert.py` | Train BERT base and SciBERT on the full Silver corpus and evaluate zero-shot transfer on IEA. | `all_results_output/40_results_bert_vs_scibert/metrics_comparison.csv` |
| `07_deeplearning/41_scibert_master.py` | Train the main SciBERT model on Silver, evaluate on IEA, save the model, and generate diagnostic figures. | `all_results_output/41_scibert_final_model/final/` and `figure*.png` |
| `07_deeplearning/42_scibert_fewshot.py` | Load the saved SciBERT model, fine-tune on 50 IEA examples, and evaluate on the remaining IEA examples. | `all_results_output/scibert_fewshot_output/` |
| `07_deeplearning/43_deep_bert_scibert_full_fewshot.py` | Run the complete BERT/SciBERT zero-shot and few-shot comparison. | `all_results_output/43_deep_all_results/metrics_deep_all_scenarios.csv` |

Commands:

```bash
python 07_deeplearning/40_bert_vs_scibert.py
python 07_deeplearning/41_scibert_master.py
python 07_deeplearning/42_scibert_fewshot.py
python 07_deeplearning/43_deep_bert_scibert_full_fewshot.py
```

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

Additional observed outputs:

- `40_bert_vs_scibert.py`: BERT IEA relaxed `59.6%`, MAE `1.84`; SciBERT IEA relaxed `54.3%`, MAE `2.08`.
- `41_scibert_master.py`: saved `41_scibert_final_model/final/`; IEA evaluation before the final figure step gave strict `17.2%`, relaxed `40.9%`, macro-F1 `0.110`.
- `42_scibert_fewshot.py`: after fine-tuning on 50 IEA examples, strict `34.8%`, relaxed `56.0%`, macro-F1 `0.162`.

## Citation

If you use this repository, please cite the associated ICPR 2026 paper. The
final citation will be added after publication.

```bibtex
@inproceedings{icpr2026_trl_ordinal_domain_shift,
  title = {Frugal and Deep Learning Approaches for Text-Based TRL Prediction under Domain Shift and Ordinal Constraints},
  author = {Anonymous},
  booktitle = {Proceedings of the International Conference on Pattern Recognition},
  year = {2026}
}
```
