import subprocess
import tempfile
import os
from utils.cfg.ast import Node

def _generate_prover9_input(premises: list[Node], conclusion: Node) -> str:
    """
    Generates a Prover9 input string from given premises and conclusion.

    Args:
        premises (list[Node]): List of premise formulas in FOL.
        conclusion (Node): Conclusion formula in FOL.

    Returns:
        str: Formatted Prover9 input string.
    """
    prover9_input = []
    
    #Header section
    prover9_input.append("set(prolog_style_variables).")
    prover9_input.append("set(auto_denials).")
    prover9_input.append("clear(print_initial_clauses).")
    prover9_input.append("clear(print_kept).")
    prover9_input.append("clear(print_given).")
    prover9_input.append("")
    
    # Premises section
    prover9_input.append("formulas(assumptions).")
    for premise in premises:
        prover9_formula = premise.to_prover9()
        prover9_input.append(f"  {prover9_formula}.")
    prover9_input.append("end_of_list.")
    prover9_input.append("")
    
    # Goal section
    prover9_input.append("formulas(goals).")
    prover9_goal = conclusion.to_prover9()
    prover9_input.append(f"  {prover9_goal}.")
    prover9_input.append("end_of_list.")
    
    return "\n".join(prover9_input)

def _run_prover9(input: str, prover9_path: str, timeout: int=30) ->bool:
    """Run the prover9 command line tool."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as temp_file:
        temp_file.write(input)
        temp_filename = temp_file.name

    try:
        result = subprocess.run(
            [prover9_path, '-f', temp_filename],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = "THEOREM PROVED" in result.stdout
    except subprocess.TimeoutExpired:
        success = False
        
    os.unlink(temp_filename)
    return success
    

def check_logical_entailment(premises: list[Node], conclusion: Node, prover9_path: str) ->bool:
    """Checks if a conclusion entails from the defined premises by using prover9."""

    prover9_input = _generate_prover9_input(premises, conclusion)
    success = _run_prover9(prover9_input, prover9_path)
    return success