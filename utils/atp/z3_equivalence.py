from utils.cfg.ast import Node
from z3 import Solver, unsat, set_param, Not


def formulas_are_equivalent(formula1: Node, formula2: Node, timeout: int=10000) -> bool:
    """Checks if two formulas are equivalent by using z3 solver by verifying the unsatisfiability
    of ¬(φ ↔ ψ), where φ is the formula produced by the LLM and ψ is the ground-truth formula.
    """

    phi = formula1.to_z3()
    psi = formula2.to_z3()

    solver = Solver()
    solver.set("timeout", timeout)
    solver.set("random_seed", 42)
    solver.add(Not(phi==psi))

    result = solver.check()
    if result == unsat:
        return True
    
    return False