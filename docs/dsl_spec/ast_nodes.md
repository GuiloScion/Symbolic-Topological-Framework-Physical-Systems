Summary:
This module defines the Abstract Syntax Tree (AST) classes used to represent structured code once tokenized and parsed. Each class models a different construct in the DSL — such as variable definitions, expressions, operations, equations, and boundary/symmetry constraints. These nodes are the compiler's internal data model — every later step (type-checking, compilation, optimization) operates on these objects.

Design Rationale:
By defining clear Python classes for each language concept, the compiler gains:

Clarity: Each node semantically represents part of the DSL.

Extensibility: Adding new DSL constructs is as simple as defining new AST classes.

Pattern Matching: Enables simple isinstance checks in later compilation stages.

Functionality Overview:
Expression: Abstract base class.

NumberExpr, IdentExpr: Leaves of expression tree.

BinaryOpExpr, UnaryOpExpr: Recursive expression structures.

VarDef: Variable declarations with type and unit.

Op: Operator with arguments.

Define: Represents function definition: LHS operator = RHS expression.

Boundary, Symmetry: Domain-specific constraints.




Code:

# === Enhanced AST Nodes ===

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
