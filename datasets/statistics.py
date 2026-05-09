import json
import argparse
import re


def print_stats(paths: list[str]) -> None:
    data = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as file:
            data += json.load(file)
    
    preds = set()
    constants = set()
    functions = set()
    
    avg_nl_length = 0
    avg_fol_length = 0
    avg_exists = 0
    avg_alls = 0
    avg_implications = 0
    avg_biimplications = 0
    avg_ands = 0
    avg_ors = 0
    avg_xors = 0
    avg_nots = 0
    avg_equals = 0
    avg_not_equals = 0
    avg_gt = 0
    avg_ge = 0
    avg_lt = 0
    avg_le = 0
    avg_plus = 0
    avg_minus = 0
    avg_mult = 0
    avg_div = 0
    
    n = len(data)
    
    for entry in data:
        nl = entry["NL"]
        formula = entry["FOL"]
        
        avg_nl_length = avg_nl_length + len(nl)
        avg_fol_length = avg_fol_length + len(formula)
        
        avg_exists = avg_exists + formula.count("∃")
        avg_alls = avg_alls + formula.count("∀")
        avg_implications = avg_implications + formula.count("→")
        avg_biimplications = avg_biimplications + formula.count("↔")
        avg_ands = avg_ands + formula.count("∧")
        avg_ors = avg_ors + formula.count("∨")
        avg_xors = avg_xors + formula.count("⊕")
        avg_nots = avg_nots + formula.count("¬")
        avg_equals = avg_equals + formula.count("=")
        avg_not_equals = avg_not_equals + formula.count("≠")
        avg_gt = avg_gt + formula.count(">")
        avg_ge = avg_ge + formula.count("≥")
        avg_lt = avg_lt + formula.count("<")
        avg_le = avg_le + formula.count("≤")
        avg_plus = avg_plus + formula.count("+")
        avg_minus = avg_minus + formula.count("-")
        avg_mult = avg_mult + formula.count("*")
        avg_div = avg_div + formula.count("/")
        
        names = re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(', formula)
        for name in names:
            if name[0].isupper():
                preds.add(name)
            else:
                functions.add(name)
        
        inside_parentheses = re.findall(r'\(([^()]*)\)', formula)
        for content in inside_parentheses:
            tokens = re.findall(r'\b[A-Za-z0-9]+\b', content)
            for t in tokens:
                if t in preds or t in functions:
                    continue
                if len(t) == 1 and t.isalpha():
                    continue
                constants.add(t)
    
    print("="*80)
    print(f"Unique predicates: {len(preds)}")
    print(f"Unique functions: {len(functions)}")
    print(f"Unique constants: {len(constants)}")
    print(f"Avg. NL length (chars): {avg_nl_length / n:.2f}")
    print(f"Avg. FOL length (chars): {avg_fol_length / n:.2f}")
    print(f"Avg. number of '∃' per formula: {avg_exists / n:.2f}")
    print(f"Avg. number of '∀' per formula: {avg_alls / n:.2f}")
    print(f"Avg. number of '→' per formula: {avg_implications / n:.2f}")
    print(f"Avg. number of '↔' per formula: {avg_biimplications / n:.2f}")
    print(f"Avg. number of '∧' per formula: {avg_ands / n:.2f}")
    print(f"Avg. number of '∨' per formula: {avg_ors / n:.2f}")
    print(f"Avg. number of '⊕' per formula: {avg_xors / n:.2f}")
    print(f"Avg. number of '¬' per formula: {avg_nots / n:.2f}")
    print(f"Total number of '=': {avg_equals}")
    print(f"Total number of '≠': {avg_not_equals}")
    print(f"Total number of '>': {avg_gt}")
    print(f"Total number of '≥': {avg_ge}")
    print(f"Total number of '<': {avg_lt}")
    print(f"Total number of '≤': {avg_le}")
    print(f"Total number of '+': {avg_plus}")
    print(f"Total number of '-': {avg_minus}")
    print(f"Total number of '*': {avg_mult}")
    print(f"Total number of '/': {avg_div}")
    print("="*30)
    longest = max(preds, key=len)
    print(longest)
    
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help="Paths to the JSON datasets"
    )
    
    args = parser.parse_args()
    
    print_stats(args.paths)

if __name__ == "__main__":
    main()