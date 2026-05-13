"""Print evaluation metrics for all (model, variant) predictions in a goldstandard JSON file.

Reads a JSON file produced by calculate_metrics.py and prints / writes a
metrics table. Each row corresponds to a (model, variant) pair labelled
"<model>__<variant>" (e.g. "llama_3.1_8b__fine-tuned"). Models that have
only one variant ("initial") are simply labelled by the model name.

Usage:
    python print_metrics.py path/to/results.json
"""

import argparse
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

RESERVED_KEYS = {"NL", "FOL"}

METRIC_COLUMNS = [
    "exact_match",
    "predicate_matched_exact_match",
    "equivalent",
    "predicate_matched_equivalent",
]

METRIC_TO_FIELD = {
    "exact_match": "EXACT_MATCH",
    "predicate_matched_exact_match": "PREDICATE_MATCHED_EXACT_MATCH",
    "equivalent": "EQUIVALENT",
    "predicate_matched_equivalent": "PREDICATE_MATCHED_EQUIVALENT",
}


def discover_model_variants(data: list[dict]) -> list[tuple[str, str]]:
    """Discover all (model, variant) pairs in the dataset, preserving order.

    Returns a list of (model_key, variant_name) tuples.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in data:
        for key, value in entry.items():
            if key in RESERVED_KEYS or not isinstance(value, dict):
                continue
            for variant_name, variant in value.items():
                if not isinstance(variant, dict):
                    continue
                pair = (key, variant_name)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    return pairs


def label_for(model: str, variant: str, single_variant_models: set[str]) -> str:
    """Build a row label. Models with a single 'initial' variant get a bare label."""
    if model in single_variant_models and variant == "initial":
        return model
    return f"{model}__{variant}"


def compute_metrics(
    entries: list[dict], pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Compute a metrics dataframe indexed by (model, variant) labels.

    Each cell is the share of entries for which that (model, variant) reports
    True for the given metric. Entries that don't contain the (model, variant)
    pair are skipped from both numerator and denominator for that row.
    """
    # Models that only ever appear with the "initial" variant get clean labels.
    variant_counts: dict[str, set[str]] = {}
    for model, variant in pairs:
        variant_counts.setdefault(model, set()).add(variant)
    single_variant_models = {
        m for m, vs in variant_counts.items() if vs == {"initial"}
    }

    rows = []
    index = []
    for model, variant in pairs:
        present = [
            entry[model][variant]
            for entry in entries
            if model in entry
            and isinstance(entry[model], dict)
            and variant in entry[model]
            and isinstance(entry[model][variant], dict)
        ]
        n = len(present)
        if n == 0:
            row = {col: float("nan") for col in METRIC_COLUMNS}
        else:
            row = {
                col: sum(1 for v in present if v.get(METRIC_TO_FIELD[col])) / n
                for col in METRIC_COLUMNS
            }
        rows.append(row)
        index.append(label_for(model, variant, single_variant_models))

    df = pd.DataFrame(rows, index=index, columns=METRIC_COLUMNS).astype(float)
    df.index.name = "model"
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print and export evaluation metrics for all (model, variant) "
            "predictions in a goldstandard JSON file."
        )
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path to the JSON file produced by calculate_metrics.py, "
            "enriched with EXACT_MATCH, EQUIVALENT, and related metric flags."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL). Default: INFO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = discover_model_variants(data)
    logger.info(
        "Discovered %d (model, variant) pairs across %d entries",
        len(pairs),
        len(data),
    )

    print("=" * 80)
    print(f"RESULTS (n = {len(data)})")
    print("=" * 80)

    df = compute_metrics(data, pairs)
    print(df.style.format(precision=4).to_string())
    print()
    print(df.style.format(precision=4).to_latex())

    base, _ = os.path.splitext(args.input)
    md_path = f"{base}_table.md"
    df.to_markdown(md_path, index=True, floatfmt=".2%")
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()