import json
import argparse
import logging

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from unicode_fol_kit import MSFLParser, Not, check_logical_entailment

logger = logging.getLogger(__name__)


def restore_original_predictions(data: list[dict], prediction_keys: list[str]) -> None:
    for entry in data:
        for key in prediction_keys:
            value = entry.get(key)
            if isinstance(value, dict) and "FOL_PRED" in value:
                entry[key] = value["FOL_PRED"]


def parse_formula(parser: MSFLParser, formula: str):
    try:
        return parser.parse(formula)
    except Exception as exc:
        logger.warning("Failed to parse formula %r: %s", formula, exc)
        return None


def find_conclusion_entry(group: list[dict]) -> dict | None:
    for entry in group:
        if entry.get("CONCLUSION") == "True":
            return entry
    return None


def determine_entailment_label(
    premises: list,
    conclusion_node,
    negated_conclusion_node,
    prover9_path: str,
) -> str:
    if check_logical_entailment(premises, conclusion_node, prover9_path):
        return "True"
    if check_logical_entailment(premises, negated_conclusion_node, prover9_path):
        return "False"
    return "Unknown"


def evaluate_group(
    group: list[dict],
    conclusion_entry: dict,
    prediction_key: str,
    prover9_path: str,
    parser: MSFLParser,
) -> str:
    premise_entries = [e for e in group if e is not conclusion_entry]

    premises = []
    skipped = 0
    for entry in premise_entries:
        formula = entry.get(prediction_key)
        if not isinstance(formula, str):
            logger.warning("Entry ID=%s has no string formula for %s", entry.get("ID"), prediction_key)
            skipped += 1
            continue

        node = parse_formula(parser, formula)
        if node is None:
            skipped += 1
            continue
        premises.append(node)

    if skipped:
        logger.info(
            "Group ID=%s, %s: skipped %d/%d unparseable premises",
            conclusion_entry.get("ID"),
            prediction_key,
            skipped,
            len(premise_entries),
        )

    if not premises:
        logger.warning(
            "Group ID=%s, %s: no parseable premises remain",
            conclusion_entry.get("ID"),
            prediction_key,
        )
        return "Unknown"

    conclusion_str = conclusion_entry.get(prediction_key)
    if not isinstance(conclusion_str, str):
        logger.warning(
            "Conclusion entry ID=%s has no string formula for %s",
            conclusion_entry.get("ID"),
            prediction_key,
        )
        return "Unknown"

    conclusion_node = parse_formula(parser, conclusion_str)
    if conclusion_node is None:
        return "Unknown"

    negated_conclusion_node = Not(conclusion_node)

    return determine_entailment_label(
        premises, conclusion_node, negated_conclusion_node, prover9_path
    )


def process_group(
    group_id: int, group: list[dict], prover9_path: str, prediction_keys: list[str]
) -> tuple[int, dict[str, str]]:
    conclusion_entry = find_conclusion_entry(group)
    if conclusion_entry is None:
        logger.warning("Group ID=%s has no entry with CONCLUSION='True' — skipping", group_id)
        return group_id, {}

    parser = MSFLParser()
    labels: dict[str, str] = {}
    for prediction_key in prediction_keys:
        label = evaluate_group(group, conclusion_entry, prediction_key, prover9_path, parser)
        labels[prediction_key] = label
        logger.info("Group ID=%s, %s -> %s", group_id, prediction_key, label)

    return group_id, labels


def compute_metrics(input_path: str, prover9_path: str, nproc: int) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    prediction_keys = sorted(k for k in data[0] if k.endswith("_pred"))
    restore_original_predictions(data, prediction_keys)

    groups: dict[int, list[dict]] = defaultdict(list)
    for entry in data:
        groups[entry["ID"]].append(entry)

    logger.info("Processing %d groups with %d worker(s)", len(groups), nproc)

    results: dict[int, dict[str, str]] = {}
    if nproc <= 1:
        for group_id, group in groups.items():
            gid, labels = process_group(group_id, group, prover9_path, prediction_keys)
            results[gid] = labels
    else:
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = {
                executor.submit(process_group, group_id, group, prover9_path, prediction_keys): group_id
                for group_id, group in groups.items()
            }
            for future in as_completed(futures):
                group_id = futures[future]
                try:
                    gid, labels = future.result()
                    results[gid] = labels
                except Exception as exc:
                    logger.error("Group ID=%s failed: %s", group_id, exc)
                    results[group_id] = {}

    for group_id, group in groups.items():
        labels = results.get(group_id, {})
        for prediction_key in prediction_keys:
            label = labels.get(prediction_key, "Unknown")
            for entry in group:
                formula = entry.get(prediction_key)
                if isinstance(formula, str):
                    entry[prediction_key] = {"FOL_PRED": formula, "ENTAILS": label}

    with open(input_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate FOLIO entailment metrics with prover9.")
    parser.add_argument("input_path", help="Path to the FOLIO results JSON file.")
    parser.add_argument("--prover9_path", type=str, required=True)
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    compute_metrics(args.input_path, args.prover9_path, args.nproc)


if __name__ == "__main__":
    main()
