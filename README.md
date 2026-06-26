# NLFOL: Advancing Natural Language Formalisation to First-Order Logic with Fine-tuned LLMs

This repository contains the code, datasets, prompts, grammar and evaluation tools used in our journal article *Advancing Natural Language Formalisation to First Order Logic with Fine-tuned LLMs*. It supports fine-tuning and evaluation of six open-source language models (T5-base, T5-3B, Flan-T5-XXL, LLaMA3.1-8B, Mistral-24B, Olmo-32B) for the translation of natural language statements into first-order predicate logic (FOL).

The fine-tuned model weights are available [here](https://huggingface.co/collections/fvossel/nl-to-fol).
The refined **GROVES** dataset (49,870 NL-FOL pairs) and the human-annotated benchmark are available [here](https://huggingface.co/datasets/fvossel/groves).

The training corpora combine the [MALLS](https://arxiv.org/abs/2305.15541) and [Willow](https://open.metu.edu.tr/handle/11511/109445) datasets. If you use these datasets, please make sure to cite the respective works.

---

## Getting Started

To get a local copy of the project up and running, execute the following commands in your terminal:

```bash
git clone https://github.com/fvossel/NLFOL
cd NLFOL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This sets up a Python virtual environment and installs all required dependencies.

## Repository Layout

- `training/` — one fine-tuning script per model (`t5_base.py`, `t5_3b.py`, `t5_11b.py`, `llama3_1_8B.py`, `mistral_24b.py`, `olmo_32b.py`) plus the LoRA configurations `lora_config.json` (decoder-only) and `lora_config_t5.json` (Flan-T5-XXL).
- `inference/` — one inference script per model, plus `logicllama.py` for the LogicLLaMA baseline.
- `datasets_nl_fol/` — train/val/test splits for the original MALLS/Willow corpus and the refined GROVES corpus, each in three variants: plain, with gold predicate lists, and with noisy predicate lists. Also contains `folio.json`, the human-annotated `goldstandard.json`, the multilingual `malls_willow_train_german_french.json`, and `statistics.py` for dataset statistics.
- `prompts/` — system prompts for the different settings: `standard_t5`, `standard_decoder`, `predicate_extraction_t5`, `predicate_extraction_decoder`, `groves_generation` (Gemini regeneration), `groves_correction` (gpt-oss-120b correction pass).
- `evaluation/` — metric computation and reporting: `calculate_metrics.py`, `print_metrics.py` (exact match + Z3 logical equivalence) and the FOLIO-specific counterparts `calculate_metrics_folio.py`, `print_metrics_folio.py` (Prover9 entailment).
- `results/` — pre-computed prediction files and Markdown summary tables for the experiments in the paper (baseline, predicate-list, predicate-list+noise, multilingual, token-extension, two-step, curriculum, FOLIO, GROVES).
- `utils/cfg/` — the Lark grammar (`syntax.lark`), parser, AST, and naming utilities used by the GROVES error-analysis pipeline (`utils/error_checking/`). Evaluation itself no longer uses these — it parses via [`unicode-fol-kit`](https://github.com/fvossel/unicode-fol-kit).
- Logical equivalence (Z3), entailment (Prover9), FOL parsing, and predicate-matched string comparison are provided by [`unicode-fol-kit`](https://github.com/fvossel/unicode-fol-kit) (a dependency, see `requirements.txt`): the evaluation scripts import `MSFLParser`, `formulas_are_equivalent`, `check_logical_entailment`, `formulas_are_identical`, `formulas_are_matched_identical`, and `match_predicates` from it.
- `utils/error_checking/` — the grammar- and spaCy-based error analysis pipeline used to produce the GROVES dataset.
- `utils/datasets/`, `utils/inference/`, `utils/training/` — shared dataset loading, batch inference, and training helpers.

## Example Usage

### Training

Train T5-base on the GROVES split with noisy predicate lists:

```bash
python -m training.t5_base \
  --system_prompt=prompts/standard_t5.txt \
  --train_path=datasets_nl_fol/groves_train_predicates_noise.json \
  --eval_path=datasets_nl_fol/groves_val_predicates_noise.json \
  --output_dir=some_path \
  --final_model=some_path
```

For bigger models that require multiple GPUs use `torchrun`:

```bash
torchrun --nproc_per_node=2 --master_port=29510 -m training.mistral_24b \
  --system_prompt=prompts/standard_decoder.txt \
  --lora_config=training/lora_config.json \
  --train_path=datasets_nl_fol/malls_willow_train_predicates_noise.json \
  --eval_path=datasets_nl_fol/malls_willow_val_predicates_noise.json \
  --output_dir=some_path \
  --final_model=some_path
```

The T5 training scripts and the decoder-only scripts also accept `--token_extension` to add dedicated tokens for the FOL connectives and quantifiers (cf. Appendix B in the paper).

### Inference

Run Flan-T5-XXL on the GROVES test set with noisy predicate lists:

```bash
python -m inference.t5_11b \
  --prefix_path=prompts/standard_t5.txt \
  --test_data_path=datasets_nl_fol/groves_test_predicates_noise.json \
  --output_file=results/groves_test_predicates_noise_t5_11b.json \
  --pred_key=t5_11b_pred \
  --adapter_path=some_path \
  --model_path=google/flan-t5-xxl
```

Decoder-only inference scripts (e.g. `inference/llama3_1_8B.py`, `inference/mistral24b.py`, `inference/olmo32b.py`) follow a slightly different argument layout: `--file_path`, `--adapter_path`, `--model_name`, `--systemprompt_path`, `--output_path`, `--pred_key`.

### Evaluation

The inference scripts write predictions back into a JSON file under a per-model key (e.g. `t5_11b_pred`). The evaluation pipeline then operates directly on those files:

**1. Exact match and logical equivalence.** `evaluation/calculate_metrics.py` enriches every prediction in place with four flags: `EXACT_MATCH`, `PREDICATE_MATCHED_EXACT_MATCH`, `EQUIVALENT`, `PREDICATE_MATCHED_EQUIVALENT`. The first two come from [`unicode-fol-kit`](https://github.com/fvossel/unicode-fol-kit) (`formulas_are_identical` / `match_predicates`: whitespace- and case-insensitive string equality, with predicate alignment via normalised Levenshtein distance with threshold 0.6); the latter two are computed by parsing both formulas with `unicode-fol-kit`'s `MSFLParser` and checking unsatisfiability of $\lnot(\varphi \leftrightarrow \psi)$ via Z3 (10-second timeout per check).

```bash
python -m evaluation.calculate_metrics results/groves_test_predicates_noise_t5_11b.json \
  --nproc=8 \
  --entry-timeout=120
```

`--nproc` controls the number of parallel workers (each with its own `CFGParser` and Z3 context), `--entry-timeout` bounds the wall-clock time per entry, and `--max-tasks-per-child` recycles workers to keep peak memory in check.

**2. Aggregate and print metrics.** `evaluation/print_metrics.py` reads the enriched JSON, prints a results table (plain text and LaTeX) to stdout and writes a Markdown table next to the input file (`<input>_table.md`):

```bash
python -m evaluation.print_metrics results/groves_test_predicates_noise_t5_11b.json
```

**3. FOLIO entailment.** For the out-of-distribution FOLIO benchmark, validity of each translated argument is checked with the Prover9 theorem prover and compared to the gold (in)validity label:

```bash
python -m evaluation.calculate_metrics_folio results/folio.json \
  --prover9_path=/path/to/prover9 \
  --nproc=8

python -m evaluation.print_metrics_folio results/folio.json results/folio_table.md
```

**4. Dataset statistics.** `datasets_nl_fol/statistics.py` reports split sizes, predicate/function/constant counts, operator frequencies, and average NL/FOL length:

```bash
python -m datasets_nl_fol.statistics \
  datasets_nl_fol/groves_train.json \
  datasets_nl_fol/groves_val.json \
  datasets_nl_fol/groves_test.json
```

Pre-computed predictions and the corresponding Markdown summary tables for all paper experiments are bundled in `results/`.

### GROVES Generation Pipeline

The refined GROVES dataset was constructed in three stages, all reproducible from this repository:

1. **Error analysis** of the combined MALLS/Willow corpus via grammar-based parsing (`utils/cfg/`) and a spaCy-based heuristic for incomplete formalisations (`utils/error_checking/error_analysis.py`).
2. **Regeneration** with Gemini 3.1 Pro using the system prompt in `prompts/groves_generation.txt`, which includes the full Lark grammar, operator-precedence rules, naming conventions, granularity guidelines, and sentence-rewriting rules for modal/adverbial constructions.
3. **Automated correction** of remaining grammar violations with a locally hosted gpt-oss-120b model (via [ollama](https://ollama.com)) using `prompts/groves_correction.txt`.

## Hardware Requirements

All experiments in the paper were conducted on a high-performance computing cluster with Intel Xeon Platinum 8480+ CPUs and either NVIDIA H100 80GB or A100-SXM4-80GB GPUs:

| Model            | Precision  | GPUs |
|------------------|------------|------|
| T5-base          | `float32`  | 1    |
| T5-3B            | `float32`  | 1    |
| Flan-T5-XXL      | `bfloat16` | 2    |
| LLaMA3.1-8B      | `bfloat16` | 1    |
| Mistral-24B      | `bfloat16` | 2    |
| Olmo-32B         | `bfloat16` | 2    |

T5-base inference runs comfortably on consumer GPUs.

## Citation

If you use this code for scientific purposes, **please cite the following paper**:

```
@misc{vossel2025advancingnaturallanguageformalization,
      title={Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs},
      author={Felix Vossel and Till Mossakowski and Björn Gehrke},
      year={2025},
      eprint={2509.22338},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22338},
}
```

## Demo

A demo of our T5-3b model can be tested here: https://translate.hyai.cs.uos.de
