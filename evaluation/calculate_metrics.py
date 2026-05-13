"""Calculate metrics for all models on a JSON dataset.

The JSON is expected to be a list of entries with at least:
    - "NL": str
    - "FOL": str  (gold standard formula)
    - one or more model keys, each mapping to a dict of variants
      (e.g. "initial", "fine-tuned"), each variant being a dict containing
      at least "PRED" with the predicted formula.

Reserved top-level keys that are NOT treated as model keys:
    NL, FOL

A model key whose value is a plain string (not a dict) is normalised to
{"initial": {"PRED": <string>}} on the fly so it gets evaluated like any
other model.

Results are written back into each variant dict as:
    EXACT_MATCH, PREDICATE_MATCHED_EXACT_MATCH,
    EQUIVALENT, PREDICATE_MATCHED_EQUIVALENT

The input file is overwritten in place.
"""

import argparse
import json
import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from tqdm import tqdm

from utils.cfg.cfgparser import CFGParser
from utils.formula_matches.match import (
    formulas_are_identical,
    formulas_are_matched_identical,
    match_predicates,
)
from utils.atp.z3_equivalence import formulas_are_equivalent

logger = logging.getLogger(__name__)

RESERVED_KEYS = {"NL", "FOL"}

_PARSER: CFGParser | None = None


def _init_worker() -> None:
    """Initialise per-worker state: disable Z3 internal threading and create a parser."""
    global _PARSER
    from z3 import set_param
    set_param("parallel.enable", False)
    _PARSER = CFGParser()


def _get_parser() -> CFGParser:
    """Return the worker-local parser, creating it on demand."""
    global _PARSER
    if _PARSER is None:
        _PARSER = CFGParser()
    return _PARSER


def normalise_entry(entry: dict) -> dict:
    """Normalise a single entry so every model key maps to a dict-of-variants."""
    for key, value in list(entry.items()):
        if key in RESERVED_KEYS:
            continue
        if isinstance(value, str):
            entry[key] = {"initial": {"PRED": value}}
    return entry


def discover_model_keys(data: list[dict]) -> list[str]:
    """Find all model-prediction keys across the dataset, preserving order."""
    found: list[str] = []
    seen: set[str] = set()
    for entry in data:
        for key in entry:
            if key in RESERVED_KEYS or key in seen:
                continue
            seen.add(key)
            found.append(key)
    return found


def evaluate_prediction(
    prediction: str,
    gold_formula: str,
    gold_formula_parsed: Any,
    parser: CFGParser,
) -> tuple[bool, bool, bool, bool]:
    """Evaluate a single prediction against the gold standard formula."""
    identical = formulas_are_identical(prediction, gold_formula)
    predicate_matched_identical = (
        True if identical else formulas_are_matched_identical(prediction, gold_formula)
    )

    if identical:
        return True, True, True, True

    if predicate_matched_identical:
        return identical, True, False, True

    try:
        prediction_parsed = parser.parse(prediction)
        equivalent = formulas_are_equivalent(prediction_parsed, gold_formula_parsed)
        if equivalent:
            return identical, predicate_matched_identical, True, True
        matched_prediction = match_predicates(prediction, gold_formula)
        matched_parsed = parser.parse(matched_prediction)
        predicate_matched_equivalent = formulas_are_equivalent(
            matched_parsed, gold_formula_parsed
        )
        return identical, predicate_matched_identical, equivalent, predicate_matched_equivalent
    except Exception:
        return identical, predicate_matched_identical, False, False


def process_entry(entry: dict) -> dict:
    """Evaluate every (model, variant) prediction inside a single entry."""
    parser = _get_parser()
    entry = normalise_entry(entry)

    gold_formula = entry["FOL"]
    try:
        gold_formula_parsed = parser.parse(gold_formula)
    except Exception:
        logger.warning("Could not parse gold formula: %s", gold_formula)
        gold_formula_parsed = None

    for key, value in entry.items():
        if key in RESERVED_KEYS or not isinstance(value, dict):
            continue
        for variant_name, variant in value.items():
            if not isinstance(variant, dict) or "PRED" not in variant:
                continue
            prediction = variant["PRED"]
            if gold_formula_parsed is None:
                identical = pm_identical = equivalent = pm_equivalent = False
            else:
                identical, pm_identical, equivalent, pm_equivalent = evaluate_prediction(
                    prediction, gold_formula, gold_formula_parsed, parser
                )
            variant["EXACT_MATCH"] = identical
            variant["PREDICATE_MATCHED_EXACT_MATCH"] = pm_identical
            variant["EQUIVALENT"] = equivalent
            variant["PREDICATE_MATCHED_EQUIVALENT"] = pm_equivalent

    return entry


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate evaluation metrics for all (model, variant) predictions "
            "in a goldstandard JSON file. Results are written back in place."
        )
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path to the JSON file containing the dataset. "
            "Must be a list of objects with at least the keys 'NL' (natural language sentence) "
            "and 'FOL' (gold-standard formula). Additional top-level keys are treated as model "
            "names. The file is overwritten in place with the computed metric fields "
            "(EXACT_MATCH, PREDICATE_MATCHED_EXACT_MATCH, EQUIVALENT, "
            "PREDICATE_MATCHED_EQUIVALENT) added to every prediction variant."
        ),
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes. Each worker maintains its own "
            "CFGParser instance and Z3 context. Use 1 (default) for sequential "
            "execution, which is easier to debug. Higher values can speed up "
            "large datasets on multi-core machines."
        ),
    )
    parser.add_argument(
        "--entry-timeout",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help=(
            "Maximum wall-clock time (in seconds) allowed for evaluating a single "
            "dataset entry when running with --nproc > 1. Entries that exceed this "
            "limit are cancelled and kept with their original content unchanged. "
            "Increase this value if your formulas are complex and Z3 equivalence "
            "checks are slow. Default: 120."
        ),
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Number of tasks a worker process handles before being replaced by a "
            "fresh one. Recycling workers bounds cumulative memory growth caused by "
            "Z3 internal state. Lower values reduce peak memory at the cost of "
            "slightly higher process-spawn overhead. Default: 100."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help=(
            "Logging verbosity level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL "
            "(case-sensitive). DEBUG prints every formula being evaluated; INFO "
            "shows progress summaries; WARNING and above suppress most output. "
            "Default: INFO."
        ),
    )
    return parser.parse_args()


def _run_parallel(
    data: list[dict],
    results: list[dict],
    nproc: int,
    entry_timeout: float,
    max_tasks_per_child: int,
) -> tuple[int, int]:
    """Run process_entry across a process pool with sliding-window submission.

    Keeps roughly 2 * nproc futures in flight at any time to avoid pre-submitting
    every entry. Writes successful results back into ``results`` by index and
    returns ``(n_timeouts, n_errors)``.
    """
    n_timeouts = 0
    n_errors = 0
    n_total = len(data)
    window_size = max(2 * nproc, 4)

    pending = iter(enumerate(data))
    in_flight: dict = {}

    def submit_next(executor: ProcessPoolExecutor) -> bool:
        """Submit the next entry, if any. Returns True if something was submitted."""
        try:
            i, entry = next(pending)
        except StopIteration:
            return False
        in_flight[executor.submit(process_entry, entry)] = i
        return True

    with ProcessPoolExecutor(
        max_workers=nproc,
        initializer=_init_worker,
        max_tasks_per_child=max_tasks_per_child,
    ) as executor:
        for _ in range(window_size):
            if not submit_next(executor):
                break

        with tqdm(total=n_total) as pbar:
            while in_flight:
                done, _not_done = wait(
                    in_flight,
                    timeout=entry_timeout,
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    oldest = next(iter(in_flight))
                    i = in_flight.pop(oldest)
                    oldest.cancel()
                    logger.warning(
                        "entry %d timed out after %.1fs, keeping original",
                        i, entry_timeout,
                    )
                    n_timeouts += 1
                    pbar.update(1)
                    submit_next(executor)
                    continue

                for fut in done:
                    i = in_flight.pop(fut)
                    try:
                        results[i] = fut.result()
                    except Exception as exc:
                        logger.error("entry %d failed: %s", i, exc)
                        n_errors += 1
                    pbar.update(1)
                    submit_next(executor)

    return n_timeouts, n_errors


def main() -> None:
    """Entry point: load JSON, dispatch evaluation, write results back in place."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_keys = discover_model_keys(data)
    logger.info("Discovered %d model keys: %s", len(model_keys), model_keys)
    logger.info(
        "Processing %d entries with nproc=%d, entry_timeout=%.1fs",
        len(data), args.nproc, args.entry_timeout,
    )

    results: list[dict] = list(data)
    n_timeouts = 0
    n_errors = 0

    if args.nproc <= 1:
        _init_worker()
        for i, entry in enumerate(tqdm(data)):
            try:
                results[i] = process_entry(entry)
            except Exception as exc:
                logger.error("entry %d failed: %s", i, exc)
                n_errors += 1
    else:
        n_timeouts, n_errors = _run_parallel(
            data,
            results,
            nproc=args.nproc,
            entry_timeout=args.entry_timeout,
            max_tasks_per_child=args.max_tasks_per_child,
        )

    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(
        "Done. Wrote %s (timeouts=%d, errors=%d)",
        args.input, n_timeouts, n_errors,
    )
    if n_timeouts or n_errors:
        print(f"[summary] timeouts={n_timeouts} errors={n_errors}")


if __name__ == "__main__":
    main()