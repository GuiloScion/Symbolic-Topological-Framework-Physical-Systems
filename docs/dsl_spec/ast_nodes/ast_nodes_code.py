class ASTNode:
    pass

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

class NumberExpr(Expression):
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Num({self.value})"

class IdentExpr(Expression):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Id({self.name})"

class BinaryOpExpr(Expression):
    def __init__(self, left: Expression, operator: str, right: Expression):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return f"BinOp({self.left} {self.operator} {self.right})"

class UnaryOpExpr(Expression):
    def __init__(self, operator: str, operand: Expression):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.operator}{self.operand})"

class VarDef(ASTNode):
    def __init__(self, name: str, vartype: str, unit: str):
        self.name = name
        self.vartype = vartype
        self.unit = unit

    def __repr__(self):
        return f"VarDef(name='{self.name}', vartype='{self.vartype}', unit='{self.unit}')"

class Op(ASTNode):
    def __init__(self, name: str, args: List[str]):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Op(name='{self.name}', args={self.args})"

class Define(ASTNode):
    def __init__(self, lhs: Op, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"Define(lhs={self.lhs}, rhs={self.rhs})"

class Boundary(ASTNode):
    def __init__(self, expr: str):
        self.expr = expr

    def __repr__(self):
        return f"Boundary(expr='{self.expr}')"

class Symmetry(ASTNode):
    def __init__(self, law: str, invariant: str):
        self.law = law
        self.invariant = invariant

    def __repr__(self):
        return f"Symmetry(law='{self.law}', invariant='{self.invariant}')"
