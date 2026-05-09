from re import sub, findall
from Levenshtein import distance

def formulas_are_identical(prediction: str, gt_formula: str) -> bool:
    """Checks if two formulas are identical, ignoring whitespace differences."""
    cleaned_formula1 = sub(r'\s+', '', prediction).lower()
    cleaned_formula2 = sub(r'\s+', '', gt_formula).lower()
    return cleaned_formula1 == cleaned_formula2

def formulas_are_matched_identical(prediction: str, gt_formula: str) -> bool:
    """Checks if two formulas are identical after matching predicates, ignoring whitespace differences."""
    matched_formula1 = match_predicates(prediction, gt_formula)
    return formulas_are_identical(matched_formula1, gt_formula)


def _map_predicates(predicate_list1: list, predicate_list2: list, max_norm_distance: float=0.6) -> list:
    """Maps predicates from predicate_list1 to predicate_list2 based on normalized Levenshtein distance."""
    mapped_list = []
    for predicate1 in predicate_list1:
        best_match = min(predicate_list2, key=lambda predicate_2: distance(predicate1, predicate_2) / max(len(predicate1), len(predicate_2)))
        norm_distance = distance(predicate1, best_match) / max(len(predicate1), len(best_match))
        
        if norm_distance <= max_norm_distance:
            mapped_list.append(best_match)
        else:
            mapped_list.append(predicate1)
    return mapped_list

def match_predicates(prediction: str, gt_formula: str) -> str:
    """Replaces predicates in formula1 according to the provided predicate_map."""
    matched_formula = prediction
    predicate_list1 = findall(r'\b\w+(?=\()', prediction)
    predicate_list2 = findall(r'\b\w+(?=\()', gt_formula)

    if len(predicate_list1) != 0 and len(predicate_list2) != 0:
        mapped_predicates = _map_predicates(predicate_list1, predicate_list2)

        for old_pred, new_pred in zip(predicate_list1, mapped_predicates):
            matched_formula = matched_formula.replace(old_pred + "(", new_pred + "(")

    return matched_formula