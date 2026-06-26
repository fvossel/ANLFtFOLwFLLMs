from unicode_fol_kit import Variable, Quantifier, Node

class ScopeAnalyzer:
    """Analyzes variable scope in AST to detect unbound variables.
    
    Maintains a stack of bound variables as the AST is traversed.
    Raises an exception if a variable is referenced outside its binding scope.
    """
    
    def __init__(self):
        """Initialize the scope analyzer with an empty bound variable stack."""
        self.bound_stack = []

    def visit(self, node):
        """Visit a node and dispatch to the appropriate visitor method.
        
        Args:
            node: The AST node to visit.
            
        Returns:
            The result of the specific visitor method.
        """
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Visit all fields of a node by default.
        
        Args:
            node: The AST node to visit.
        """
        for field in getattr(node, "__dataclass_fields__", {}).values():
            value = getattr(node, field.name)

            if isinstance(value, list):
                for v in value:
                    self.visit(v)
            else:
                self.visit(value)
    
    def visit_Variable(self, node: Variable):
        """Check if a variable is bound in the current scope.
        
        Args:
            node: The Variable node to check.
            
        Raises:
            Exception: If the variable is not bound by any quantifier.
        """
        if node.name not in self.bound_stack:
            raise Exception(f"FREE_VARIABLE_ERROR: Variable '{node.name}' is unbound. Every variable must be bound by a quantifier (∀ or ∃).")
    
    def visit_Quantifier(self, node: Quantifier):
        """Handle quantifier by adding variable to scope and visiting the formula.
        
        Args:
            node: The Quantifier node (∀ or ∃).
        """
        self.bound_stack.append(node.variable.name)
        self.visit(node.formula)
        self.bound_stack.pop()
    
    def visit_Not(self, node):
        """Visit negation formula.
        
        Args:
            node: The Not node.
        """
        self.visit(node.formula)

    def visit_And(self, node):
        """Visit conjunction formula.
        
        Args:
            node: The And node.
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_Or(self, node):
        """Visit disjunction formula.
        
        Args:
            node: The Or node.
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_Implies(self, node):
        """Visit implication formula.
        
        Args:
            node: The Implies node.
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_Iff(self, node):
        """Visit biconditional formula.
        
        Args:
            node: The Iff node.
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_Xor(self, node):
        """Visit exclusive or formula.
        
        Args:
            node: The Xor node.
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_Atom(self, node):
        """Visit atom by checking all argument terms.
        
        Args:
            node: The Atom node.
        """
        for arg in node.args:
            self.visit(arg)

    def visit_Function(self, node):
        """Visit function by checking all argument terms.
        
        Args:
            node: The Function node.
        """
        for arg in node.args:
            self.visit(arg)

    def visit_Constant(self, node):
        """Visit constant (no variables to check).
        
        Args:
            node: The Constant node.
        """
        pass
    
    def visit_Number(self, node):
        """Visit number (no variables to check).
        
        Args:
            node: The Number node.
        """
        pass