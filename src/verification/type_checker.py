class TypeChecker:
    def __init__(self):
        self.variables: Dict[str, Tuple[str, Unit]] = {}
        self.operators: Dict[str, Tuple[List[str], str]] = {}

    def get_unit(self, unit_str: str) -> Unit:
        """Convert unit string to Unit object"""
        if unit_str in UNITS:
            return UNITS[unit_str]
        elif unit_str in ["1", "1.0"] or unit_str.replace(".", "").isdigit():
            # Handle numeric units as dimensionless
            return UNITS["dimensionless"]
        else:
            # For unknown units, create a custom unit
            return Unit({unit_str: 1})

    def check_expression_type(self, expr: Expression) -> Unit:
        """Check the dimensional consistency of an expression"""
        if isinstance(expr, NumberExpr):
            return UNITS["dimensionless"]

        elif isinstance(expr, IdentExpr):
            if expr.name not in self.variables:
                raise TypeError(f"Undefined variable: {expr.name}")
            _, unit = self.variables[expr.name]
            return unit

        elif isinstance(expr, BinaryOpExpr):
            left_unit = self.check_expression_type(expr.left)
            right_unit = self.check_expression_type(expr.right)

            if expr.operator == "+":
                if not left_unit.is_compatible(right_unit):
                    raise TypeError(f"Cannot add incompatible units: {left_unit} + {right_unit}")
                return left_unit

            elif expr.operator == "-":
                if not left_unit.is_compatible(right_unit):
                    raise TypeError(f"Cannot subtract incompatible units: {left_unit} - {right_unit}")
                return left_unit

            elif expr.operator == "*":
                return left_unit * right_unit

            elif expr.operator == "/":
                return left_unit / right_unit

            elif expr.operator == "^":
                if not isinstance(expr.right, NumberExpr):
                    raise TypeError("Exponent must be a number")
                return left_unit ** expr.right.value

        elif isinstance(expr, UnaryOpExpr):
            operand_unit = self.check_expression_type(expr.operand)
            if expr.operator == "-":
                return operand_unit

        raise TypeError(f"Unknown expression type: {type(expr)}")

    def check(self, ast: List[ASTNode]):
        print("\n[Enhanced TypeCheck] Starting type checking...")

        # First pass: collect variable definitions
        for node in ast:
            if isinstance(node, VarDef):
                unit = self.get_unit(node.unit)
                self.variables[node.name] = (node.vartype, unit)
                print(f"[TypeCheck] {node.name}: type = {node.vartype}, unit = {unit}")

        # Second pass: check definitions and expressions
        for node in ast:
            if isinstance(node, Define):
                if isinstance(node.lhs, Op):
                    # Check that all arguments are defined
                    for arg in node.lhs.args:
                        if arg not in self.variables:
                            raise TypeError(f"Undefined variable in operator: {arg}")

                    # Check dimensional consistency of RHS
                    try:
                        rhs_unit = self.check_expression_type(node.rhs)
                        print(f"[TypeCheck] Definition {node.lhs.name} has result unit: {rhs_unit}")
                    except Exception as e:
                        print(f"[TypeCheck] Warning in definition {node.lhs.name}: {e}")
                        # Continue processing rather than stopping on dimensional analysis errors

            elif isinstance(node, Boundary):
                if node.expr not in self.variables:
                    raise TypeError(f"Boundary references undefined variable: {node.expr}")
                print(f"[TypeCheck] Boundary condition on: {node.expr}")

            elif isinstance(node, Symmetry):
                print(f"[TypeCheck] Symmetry declared: {node.law} invariant under {node.invariant}")