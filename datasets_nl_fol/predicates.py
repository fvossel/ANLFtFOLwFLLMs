import json
import random
import re
from tqdm import tqdm

def get_pred_set(fol_formula: str) -> set[str]:
    return set(re.findall(r"\b\w+(?=\()", fol_formula))

def extract_predicates(fol_formula: str) -> set[str]:
    """Extract predicate names from a FOL formula string."""
    all_predicates = set(re.findall(r"\b\w+(?=\()", fol_formula))
    return "[" + ", ".join(sorted(all_predicates)) + "]"

def build_predicates_with_noise(example: dict, dataset, num_noise_samples: int = 5) -> str:
    """Build a predicate string including current predicates plus noise predicates from random examples."""
    current_predicates = get_pred_set(example["FOL"])
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    random_examples = [dataset[i] for i in indices[:min(num_noise_samples, len(dataset))]]

    noise_predicates = set()
    for random_example in random_examples:
        noise_predicates.update(get_pred_set(random_example["FOL"]))

    all_predicates = current_predicates.union(noise_predicates)
    return "[" + ", ".join(sorted(all_predicates)) + "]"




with open("datasets/malls_willow_train.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    

for entry in tqdm(data, desc="Processing entries", total=len(data)):
    entry["NL"] = entry["NL"] + build_predicates_with_noise(entry, data)
    

with open("datasets/malls_willow_train_predicates_noise.json", "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)