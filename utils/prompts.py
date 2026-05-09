"""Shared system prompts and prompt construction utilities for FOL translation experiments."""

import re
import random


SYSTEM_PROMPT_BASE = (
    "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) "
    "using only the given quantors and junctors:"
    "\u2200 (for all), \u2203 (there exists), \u00ac (not), \u2227 (and), \u2228 (or), \u2192 (implies), "
    "\u2194 (if and only if), \u2295 (xor)."
    "Start your answer with '\ud835\udf19=' followed by the FOL-formula. Do not include any other text."
)

SYSTEM_PROMPT_PREDICATE_EXTRACTION = (
    "You are a helpful AI assistant that extracts predicates from Natural Language (NL) text "
    "for translating into First-Order Logic (FOL):"
    "Start your answer with 'Predicates=[' followed by the predicates in alphabetical order "
    "followed by ']'. Do not include any other text."
)

T5_SYSTEM_PROMPT = "translate English natural language statements into first-order logic (FOL): "


def build_system_prompt_with_predicates(predicates_string: str) -> str:
    """Build a system prompt that includes a predicate constraint."""
    return (
        "You are a helpful AI assistant that translates Natural Language (NL) text in First-Order Logic (FOL) "
        "using only the given quantors and junctors:"
        "\u2200 (for all), \u2203 (there exists), \u00ac (not), \u2227 (and), \u2228 (or), \u2192 (implies), "
        "\u2194 (if and only if), \u2295 (xor)."
        "Use only the following predicates:" + predicates_string
        + "Start your answer with '\ud835\udf19=' followed by the FOL-formula. Do not include any other text."
    )


def extract_predicates(fol_formula: str) -> set[str]:
    """Extract predicate names from a FOL formula string."""
    return set(re.findall(r"\b\w+(?=\()", fol_formula))


def build_predicates_with_noise(example: dict, dataset, num_noise_samples: int = 5) -> str:
    """Build a predicate string including current predicates plus noise predicates from random examples."""
    current_predicates = extract_predicates(example["FOL"])
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random_examples = [dataset[i] for i in indices[:min(num_noise_samples, len(dataset))]]

    noise_predicates = set()
    for random_example in random_examples:
        noise_predicates.update(extract_predicates(random_example["FOL"]))

    all_predicates = current_predicates.union(noise_predicates)
    return ", ".join(sorted(all_predicates))


def load_system_prompt_from_file(filepath: str) -> str:
    """Load a system prompt from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
