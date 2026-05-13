"""Print evaluation metrics for all models on the FOLIO dataset."""

import argparse
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ["model", "entailment_accuracy"]


def compute_metrics(entries: list, prediction_keys: list) -> pd.DataFrame:
    seen_ids = set()
    unique_entries = []
    for entry in entries:
        eid = entry["ID"]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        unique_entries.append(entry)

    df = pd.DataFrame(columns=METRIC_COLUMNS)
    df["model"] = prediction_keys
    df = df.set_index("model")
    n = len(unique_entries)

    if n == 0:
        return df.astype(float)

    for key in prediction_keys:
        num_correct = sum(
            1
            for entry in unique_entries
            if isinstance(entry.get(key), dict)
            and entry[key].get("ENTAILS") == entry.get("LABEL")
        )
        df.loc[key, "entailment_accuracy"] = num_correct / n

    df = df.astype(float)
    df.index = df.index.str.replace("_pred", "", regex=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Print FOLIO evaluation metrics.")
    parser.add_argument("input_path", help="Path to the FOLIO results JSON file.")
    parser.add_argument("output_table_path", help="Path to write the Markdown metrics table.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prediction_keys = sorted(k for k in data[0] if k.endswith("_pred"))

    n_groups = len({entry["ID"] for entry in data})

    print("=" * 80)
    print(f"FOLIO ENTAILMENT RESULTS (groups = {n_groups}, entries = {len(data)})")
    print("=" * 80)
    df = compute_metrics(data, prediction_keys)
    print(df.style.format(precision=4).to_string())
    print()
    print(df.style.format(precision=4).to_latex())

    df.to_markdown(args.output_table_path, index=True, floatfmt=".2%")
    logger.info("Wrote metrics table to %s", args.output_table_path)


if __name__ == "__main__":
    main()