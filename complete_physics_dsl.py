import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

# ============
# TOKEN SYSTEM
# ============

TOKEN_TYPES = [
    # Physics specific -- PUT THESE FIRST!
    ("SYSTEM", r"\\system"),
    ("DEFVAR", r"\\defvar"),
    ("DEFINE", r"\\define"),
    ("LAGRANGIAN", r"\\lagrangian"),
    ("HAMILTONIAN", r"\\hamiltonian"),
    ("TRANSFORM", r"\\transform"),
    ("CONSTRAINT", r"\\constraint"),
    ("INITIAL", r"\\initial"),
    ("SOLVE", r"\\solve"),
    ("ANIMATE", r"\\animate"),
    ("PLOT", r"\\plot"),
    # Vector operations
    ("VEC", r"\\vec"),
    ("HAT", r"\\hat"),
    ("MAGNITUDE", r"\\mag|\\norm"),
    # Time derivatives
    ("DOT_NOTATION", r"\\dot"),
    ("DDOT_NOTATION", r"\\ddot"),
    # Advanced math operators
    ("VECTOR_DOT", r"\\cdot|\\dot"),
    ("VECTOR_CROSS", r"\\times|\\cross"),
    ("GRADIENT", r"\\nabla|\\grad"),
    ("DIVERGENCE", r"\\div"),
    ("CURL", r"\\curl"),
    ("LAPLACIAN", r"\\laplacian|\\Delta"),
    # Calculus
    ("PARTIAL", r"\\partial"),
    ("INTEGRAL", r"\\int"),
    ("OINT", r"\\oint"),
    ("SUM", r"\\sum"),
    ("LIMIT", r"\\lim"),
    # Greek letters (put before COMMAND so they don't get gobbled!)
    ("GREEK_LETTER", r"\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\zeta|\\eta|\\theta|\\iota|\\kappa|\\lambda|\\mu|\\nu|\\xi|\\omicron|\\pi|\\rho|\\sigma|\\tau|\\upsilon|\\phi|\\chi|\\psi|\\omega"),
    # General commands -- MUST BE AFTER ALL SPECIFIC TOKENS!
    ("COMMAND", r"\\[a-zA-Z_][a-zA-Z0-9_]*"),
    # Brackets and grouping
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    # Mathematical operators
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MULTIPLY", r"\*"),
    ("DIVIDE", r"/"),
    ("POWER", r"\^"),
    ("EQUALS", r"="),
    ("COMMA", r","),
    ("SEMICOLON", r";"),
    ("DOT", r"\."),
    ("UNDERSCORE", r"_"),
    ("PIPE", r"\|"),
    # Basic tokens
    ("NUMBER", r"\d+\.?\d*([eE][+-]?\d+)?"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("WHITESPACE", r"\s+"),
    ("NEWLINE", r"\n"),
    ("COMMENT", r"%.*"),
]

# Compile regex
token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)

@dataclass
class Token:
    type: str
    value: str
    position: int = 0
    line: int = 1
    column: int = 1

    def __repr__(self):
        return f"{self.type}:{self.value}@{self.line}:{self.column}"

def tokenize(source: str) -> List[Token]:
    """Enhanced tokenizer with position tracking"""
    tokens = []
    line = 1
    line_start = 0
    
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        
        # Update line tracking
        while line_start < position and '\n' in source[line_start:position]:
            newline_pos = source.find('\n', line_start)
            if newline_pos != -1 and newline_pos < position:
                line += 1
                line_start = newline_pos + 1
            else:
                break
                
        column = position - line_start + 1
        
        if kind not in ["WHITESPACE", "COMMENT"]:
            tokens.append(Token(kind, value, position, line, column))
            
    return tokens

# ===========================
# COMPLETE AST SYSTEM
# ===========================

class ASTNode:
    """Base class for all AST nodes"""
    pass

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

# Basic expressions
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

class GreekLetterExpr(Expression):
    def __init__(self, letter: str):
        self.letter = letter
    def __repr__(self):
        return f"Greek({self.letter})"

class DerivativeVarExpr(Expression):
    def __init__(self, var: str, order: int = 1):
        self.var = var
        self.order = order
    def __repr__(self):
        return f"DerivativeVar({self.var}, order={self.order})"

# Binary operations
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

# Vector expressions
class VectorExpr(Expression):
    def __init__(self, components: List[Expression]):
        self.components = components
    def __repr__(self):
        return f"Vector({self.components})"

class VectorOpExpr(Expression):
    def __init__(self, operation: str, left: Expression, right: Expression = None):
        self.operation = operation
        self.left = left
        self.right = right
    def __repr__(self):
        if self.right:
            return f"VectorOp({self.operation}, {self.left}, {self.right})"
        return f"VectorOp({self.operation}, {self.left})"

# Calculus expressions
class DerivativeExpr(Expression):
    def __init__(self, expr: Expression, var: str, order: int = 1, partial: bool = False):
        self.expr = expr
        self.var = var
        self.order = order
        self.partial = partial
    def __repr__(self):
        type_str = "Partial" if self.partial else "Total"
        return f"{type_str}Deriv({self.expr}, {self.var}, order={self.order})"

class IntegralExpr(Expression):
    def __init__(self, expr: Expression, var: str, lower=None, upper=None, line_integral=False):
        self.expr = expr
        self.var = var
        self.lower = lower
        self.upper = upper
        self.line_integral = line_integral
    def __repr__(self):
        return f"Integral({self.expr}, {self.var}, {self.lower}, {self.upper})"

# Function calls
class FunctionCallExpr(Expression):
    def __init__(self, name: str, args: List[Expression]):
        self.name = name
        self.args = args
    def __repr__(self):
        return f"Call({self.name}, {self.args})"

# Physics-specific AST nodes
class SystemDef(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"System({self.name})"

class VarDef(ASTNode):
    def __init__(self, name: str, vartype: str, unit: str, vector: bool = False):
        self.name = name
        self.vartype = vartype
        self.unit = unit
        self.vector = vector
    def __repr__(self):
        vec_str = " [Vector]" if self.vector else ""
        return f"VarDef({self.name}: {self.vartype}[{self.unit}]{vec_str})"

class DefineDef(ASTNode):
    def __init__(self, name: str, args: List[str], body: Expression):
        self.name = name
        self.args = args
        self.body = body
    def __repr__(self):
        return f"Define({self.name}({', '.join(self.args)}) = {self.body})"

class LagrangianDef(ASTNode):
    def __init__(self, expr: Expression):
        self.expr = expr
    def __repr__(self):
        return f"Lagrangian({self.expr})"

class HamiltonianDef(ASTNode):
    def __init__(self, expr: Expression):
        self.expr = expr
    def __repr__(self):
        return f"Hamiltonian({self.expr})"

class TransformDef(ASTNode):
    def __init__(self, coord_type: str, var: str, expr: Expression):
        self.coord_type = coord_type
        self.var = var
        self.expr = expr
    def __repr__(self):
        return f"Transform({self.coord_type}: {self.var} = {self.expr})"

class InitialCondition(ASTNode):
    def __init__(self, conditions: Dict[str, float]):
        self.conditions = conditions
    def __repr__(self):
        return f"Initial({self.conditions})"

class SolveDef(ASTNode):
    def __init__(self, method: str, options: Dict[str, Any] = None):
        self.method = method
        self.options = options or {}
    def __repr__(self):
        return f"Solve({self.method}, {self.options})"

class AnimateDef(ASTNode):
    def __init__(self, target: str, options: Dict[str, Any] = None):
        self.target = target
        self.options = options or {}
    def __repr__(self):
        return f"Animate({self.target}, {self.options})"

# ===========================
# ENHANCED PHYSICS UNITS
# ===========================

class Unit:
    def __init__(self, dimensions: Dict[str, int], scale: float = 1.0):
        self.dimensions = dimensions
        self.scale = scale

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Unit(self.dimensions, self.scale * other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) + other.dimensions.get(dim, 0)
        return Unit(result, self.scale * other.scale)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Unit(self.dimensions, self.scale / other)
        result = {}
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) - other.dimensions.get(dim, 0)
        return Unit(result, self.scale / other.scale)

    def __pow__(self, exponent):
        result = {dim: power * exponent for dim, power in self.dimensions.items()}
        return Unit(result, self.scale ** exponent)

    def is_compatible(self, other):
        return self.dimensions == other.dimensions

    def __repr__(self):
        return f"Unit({self.dimensions}, scale={self.scale})"

# Comprehensive unit system
BASE_UNITS = {
    "dimensionless": Unit({}),
    "1": Unit({}),
    
    # SI Base units
    "m": Unit({"length": 1}),
    "kg": Unit({"mass": 1}),
    "s": Unit({"time": 1}),
    "A": Unit({"current": 1}),
    "K": Unit({"temperature": 1}),
    "mol": Unit({"substance": 1}),
    "cd": Unit({"luminous_intensity": 1}),
    
    # Common derived units
    "N": Unit({"mass": 1, "length": 1, "time": -2}),  # Force
    "J": Unit({"mass": 1, "length": 2, "time": -2}),  # Energy
    "W": Unit({"mass": 1, "length": 2, "time": -3}),  # Power
    "Pa": Unit({"mass": 1, "length": -1, "time": -2}), # Pressure
    "Hz": Unit({"time": -1}),  # Frequency
    "C": Unit({"current": 1, "time": 1}),  # Charge
    "V": Unit({"mass": 1, "length": 2, "time": -3, "current": -1}),  # Voltage
    "F": Unit({"mass": -1, "length": -2, "time": 4, "current": 2}),  # Capacitance
    "Wb": Unit({"mass": 1, "length": 2, "time": -2, "current": -1}),  # Magnetic flux
    "T": Unit({"mass": 1, "time": -2, "current": -1}),  # Magnetic field
    
    # Angle units
    "rad": Unit({"angle": 1}),
    "deg": Unit({"angle": 1}, scale=np.pi/180),
    
    # Common physics units
    "eV": Unit({"mass": 1, "length": 2, "time": -2}, scale=1.602e-19),
    "c": Unit({"length": 1, "time": -1}, scale=299792458),  # Speed of light
    "hbar": Unit({"mass": 1, "length": 2, "time": -1}, scale=1.055e-34),
    "G": Unit({"mass": -1, "length": 3, "time": -2}, scale=6.674e-11),
    "k_B": Unit({"mass": 1, "length": 2, "time": -2, "temperature": -1}, scale=1.381e-23),
}

# ==============
# PARSER ENGINE
# =============

class MechanicsParser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_system = None

    def peek(self, offset: int = 0) -> Optional[Token]:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None

    def match(self, *expected_types: str) -> Optional[Token]:
        token = self.peek()
        if token and token.type in expected_types:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        token = self.match(expected_type)
        if not token:
            current = self.peek()
            if current:
                raise SyntaxError(f"Expected {expected_type} but got {current.type} '{current.value}' at {current.line}:{current.column}")
            else:
                raise SyntaxError(f"Expected {expected_type} but reached end of input")
        return token

    def parse(self) -> List[ASTNode]:
        """Parse the complete DSL"""
        nodes = []
        while self.pos < len(self.tokens):
            try:
                node = self.parse_statement()
                if node:
                    nodes.append(node)
            except Exception as e:
                current = self.peek()
                if current:
                    raise SyntaxError(f"Parse error at {current.line}:{current.column}: {e}")
                else:
                    raise SyntaxError(f"Parse error at end of input: {e}")
        return nodes

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a top-level statement"""
        token = self.peek()
        if not token:
            return None

        if token.type == "SYSTEM":
            return self.parse_system()
        elif token.type == "DEFVAR":
            return self.parse_defvar()
        elif token.type == "DEFINE":
            return self.parse_define()
        elif token.type == "LAGRANGIAN":
            return self.parse_lagrangian()
        elif token.type == "HAMILTONIAN":
            return self.parse_hamiltonian()
        elif token.type == "TRANSFORM":
            return self.parse_transform()
        elif token.type == "INITIAL":
            return self.parse_initial()
        elif token.type == "SOLVE":
            return self.parse_solve()
        elif token.type == "ANIMATE":
            return self.parse_animate()
        else:
            # Skip unknown tokens
            self.pos += 1
            return None

    def parse_system(self) -> SystemDef:
        """Parse \\system{name}"""
        self.expect("SYSTEM")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.current_system = name
        return SystemDef(name)

    def parse_defvar(self) -> VarDef:
        """Parse \\defvar{name}{type}{unit}"""
        self.expect("DEFVAR")
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        vartype = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        
        # Parse unit (could be complex)
        unit_expr = self.parse_expression()
        unit = self.expression_to_string(unit_expr)
        
        self.expect("RBRACE")
        
        # Check if it's a vector type
        is_vector = vartype in ["Vector", "Vector3", "Position", "Velocity", "Force"]
        
        return VarDef(name, vartype, unit, is_vector)

    def parse_define(self) -> DefineDef:
        """Parse \\define{\\op{name}(args) = expression}"""
        self.expect("DEFINE")
        self.expect("LBRACE")
        
        # Expect \\op{name}
        self.expect("COMMAND")  # Should be \\op
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        
        # Parse arguments
        self.expect("LPAREN")
        args = []
        if self.peek() and self.peek().type == "IDENT":
            args.append(self.expect("IDENT").value)
            while self.match("COMMA"):
                args.append(self.expect("IDENT").value)
        self.expect("RPAREN")
        
        self.expect("EQUALS")
        
        # Parse the expression
        body = self.parse_expression()
        
        self.expect("RBRACE")
        
        return DefineDef(name, args, body)

    def parse_lagrangian(self) -> LagrangianDef:
        """Parse \\lagrangian{expression}"""
        self.expect("LAGRANGIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return LagrangianDef(expr)

    def parse_hamiltonian(self) -> HamiltonianDef:
        """Parse \\hamiltonian{expression}"""
        self.expect("HAMILTONIAN")
        self.expect("LBRACE")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return HamiltonianDef(expr)

    def parse_transform(self) -> TransformDef:
        """Parse \\transform{type}{var = expr}"""
        self.expect("TRANSFORM")
        self.expect("LBRACE")
        coord_type = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        expr = self.parse_expression()
        self.expect("RBRACE")
        return TransformDef(coord_type, var, expr)

    def parse_initial(self) -> InitialCondition:
        """Parse \\initial{var1=val1, var2=val2, ...}"""
        self.expect("INITIAL")
        self.expect("LBRACE")
        
        conditions = {}
        var = self.expect("IDENT").value
        self.expect("EQUALS")
        val = float(self.expect("NUMBER").value)
        conditions[var] = val
        
        while self.match("COMMA"):
            var = self.expect("IDENT").value
            self.expect("EQUALS")
            val = float(self.expect("NUMBER").value)
            conditions[var] = val
            
        self.expect("RBRACE")
        return InitialCondition(conditions)

    def parse_solve(self) -> SolveDef:
        """Parse \\solve{method}"""
        self.expect("SOLVE")
        self.expect("LBRACE")
        method = self.expect("IDENT").value
        # TODO: Parse options
        self.expect("RBRACE")
        return SolveDef(method)

    def parse_animate(self) -> AnimateDef:
        """Parse \\animate{target}"""
        self.expect("ANIMATE")
        self.expect("LBRACE")
        target = self.expect("IDENT").value
        # TODO: Parse options
        self.expect("RBRACE")
        return AnimateDef(target)

    # Expression parsing with full precedence
    def parse_expression(self) -> Expression:
        """Parse expressions with full operator precedence"""
        return self.parse_logical()

    def parse_logical(self) -> Expression:
        """Logical operators (lowest precedence)"""
        left = self.parse_additive()
        # TODO: Add logical operators if needed
        return left

    def parse_additive(self) -> Expression:
        """Addition and subtraction"""
        left = self.parse_multiplicative()
        
        while True:
            if self.match("PLUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "+", right)
            elif self.match("MINUS"):
                right = self.parse_multiplicative()
                left = BinaryOpExpr(left, "-", right)
            else:
                break
                
        return left

    def parse_multiplicative(self) -> Expression:
        """Multiplication, division, and implicit multiplication"""
        left = self.parse_power()
        
        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            else:
                # Check for implicit multiplication
                next_token = self.peek()
                if (next_token and 
                    next_token.type in ["IDENT", "NUMBER", "LPAREN", "GREEK_LETTER"] and
                    not self.at_end_of_expression()):
                    right = self.parse_power()
                    left = BinaryOpExpr(left, "*", right)
                else:
                    break
                    
        return left

    def parse_power(self) -> Expression:
        """Exponentiation (right associative)"""
        left = self.parse_unary()
        
        if self.match("POWER"):
            right = self.parse_power()  # Right associative
            return BinaryOpExpr(left, "^", right)
            
        return left

    def parse_unary(self) -> Expression:
        """Unary operators and function calls"""
        if self.match("MINUS"):
            operand = self.parse_unary()
            return UnaryOpExpr("-", operand)
        elif self.match("PLUS"):
            return self.parse_unary()
        
        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Function calls, derivatives, etc."""
        expr = self.parse_primary()
        
        while True:
            if self.match("LPAREN"):
                # Function call
                args = []
                if not self.peek() or self.peek().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self.match("COMMA"):
                        args.append(self.parse_expression())
                self.expect("RPAREN")
                
                if isinstance(expr, IdentExpr):
                    expr = FunctionCallExpr(expr.name, args)
                else:
                    raise SyntaxError("Invalid function call syntax")
                    
            elif self.match("DOT"):
                # Method call or dot notation (like theta.dot)
                method = self.expect("IDENT").value
                if method == "dot" and isinstance(expr, IdentExpr):
                    # Convert to time derivative
                    expr = DerivativeExpr(expr, "t", 1, False)
                else:
                    # TODO: Handle other dot operations
                    pass
                    
            else:
                break
                
        return expr

    def parse_primary(self) -> Expression:
        """Primary expressions: literals, identifiers, parentheses, vectors"""
        
        # Numbers
        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        def parse_primary(self) -> Expression:
   
         # Time derivative tokens
        if self.match("DOT_NOTATION"):
        # \dot{IDENT}
           self.expect("LBRACE")
           var = self.expect("IDENT").value
           self.expect("RBRACE")
           return DerivativeVarExpr(var, 1)
        if self.match("DDOT_NOTATION"):
            # \ddot{IDENT}
           self.expect("LBRACE")
           var = self.expect("IDENT").value
           self.expect("RBRACE")
           return DerivativeVarExpr(var, 2)
        
        # Identifiers
        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)
        
        # Greek letters
        if self.match("GREEK_LETTER"):
            letter = self.tokens[self.pos - 1].value[1:]  # Remove backslash
            return GreekLetterExpr(letter)
        
        # Parentheses
        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr
        
        # Vectors [x, y, z]
        if self.match("LBRACKET"):
            components = []
            components.append(self.parse_expression())
            while self.match("COMMA"):
                components.append(self.parse_expression())
            self.expect("RBRACKET")
            return VectorExpr(components)
        
        # Commands (LaTeX-style functions)
        if self.match("COMMAND"):
            cmd = self.tokens[self.pos - 1].value
            return self.parse_command(cmd)
        
        # Mathematical constants
        if self.peek() and self.peek().value in ["pi", "e"]:
            const = self.expect("IDENT").value
            if const == "pi":
                return NumberExpr(np.pi)
            elif const == "e":
                return NumberExpr(np.e)
        
        current = self.peek()
        if current:
            raise SyntaxError(f"Unexpected token {current.type} '{current.value}' at {current.line}:{current.column}")
        else:
            raise SyntaxError("Unexpected end of input")

    def parse_command(self, cmd: str) -> Expression:
        """Parse LaTeX-style commands"""
        
        if cmd == r"\vec":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("vec", expr)
            
        elif cmd == r"\hat":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return VectorOpExpr("unit", expr)
            
        elif cmd == r"\dot":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return DerivativeExpr(expr, "t", 1, False)
            
        elif cmd == r"\ddot":
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            return DerivativeExpr(expr, "t", 2, False)
            
        elif cmd == r"\partial":
            # \partial{f}{x}
            self.expect("LBRACE")
            expr = self.parse_expression()
            self.expect("RBRACE")
            self.expect("LBRACE")
            var = self.expect("IDENT").value
            self.expect("RBRACE")
            return DerivativeExpr(expr, var, 1, True)
            
        elif cmd in [r"\sin", r"\cos", r"\tan", r"\exp", r"\log", r"\sqrt"]:
            func_name = cmd[1:]  # Remove backslash
            self.expect("LBRACE")
            arg = self.parse_expression()
            self.expect("RBRACE")
            return FunctionCallExpr(func_name, [arg])
            
        elif cmd == r"\cdot":
            # Already handled as infix operator
            return IdentExpr("cdot")
            
        elif cmd == r"\times":
            # Cross product - handled as infix
            return IdentExpr("times")
            
        elif cmd == r"\nabla":
            # Gradient operator
            if self.peek() and self.peek().type == "IDENT":
                expr = IdentExpr(self.expect("IDENT").value)
                return VectorOpExpr("grad", expr)
            return VectorOpExpr("grad", None)
            
        else:
            # Unknown command - treat as identifier
            return IdentExpr(cmd[1:])

    def at_end_of_expression(self) -> bool:
        """Check if we're at the end of an expression"""
        token = self.peek()
        return (not token or 
                token.type in ["RBRACE", "RPAREN", "RBRACKET", "COMMA", "SEMICOLON", "EQUALS"])

    def expression_to_string(self, expr: Expression) -> str:
        """Convert expression back to string for unit parsing"""
        if isinstance(expr, NumberExpr):
            return str(expr.value)
        elif isinstance(expr, IdentExpr):
            return expr.name
        elif isinstance(expr, BinaryOpExpr):
            left = self.expression_to_string(expr.left)
            right = self.expression_to_string(expr.right)
            return f"({left} {expr.operator} {right})"
        elif isinstance(expr, UnaryOpExpr):
            operand = self.expression_to_string(expr.operand)
            return f"{expr.operator}{operand}"
        else:
            return str(expr)

# ===========================
# SYMBOLIC MATH ENGINE
# ===========================

class SymbolicEngine:
    def __init__(self):
        self.sp = sp
        self.symbol_map = {}
        self.function_map = {}
        self.time_symbol = sp.Symbol('t')

    def get_symbol(self, name: str) -> sp.Symbol:
        """Get or create a SymPy symbol"""
        if name not in self.symbol_map:
            self.symbol_map[name] = sp.Symbol(name, real=True)
        return self.symbol_map[name]

    def get_function(self, name: str) -> sp.Function:
        """Get or create a SymPy function"""
        if name not in self.function_map:
            self.function_map[name] = sp.Function(name)
        return self.function_map[name]

    def ast_to_sympy(self, expr: Expression) -> sp.Expr:
        """Convert AST expression to SymPy"""
        
        if isinstance(expr, NumberExpr):
            return sp.Float(expr.value)
            
        elif isinstance(expr, IdentExpr):
            return self.get_symbol(expr.name)
            
        elif isinstance(expr, GreekLetterExpr):
            return self.get_symbol(expr.letter)
            
        elif isinstance(expr, DerivativeVarExpr):
             if expr.order == 1:
           return self.get_symbol(f"{expr.var}_dot")
        elif expr.order == 2:
            return self.get_symbol(f"{expr.var}_ddot")
        else:
            raise ValueError("Only first and second order derivatives are supported")
            
        elif isinstance(expr, BinaryOpExpr):
            left = self.ast_to_sympy(expr.left)
            right = self.ast_to_sympy(expr.right)
            
            if expr.operator == "+":
                return left + right
            elif expr.operator == "-":
                return left - right
            elif expr.operator == "*":
                return left * right
            elif expr.operator == "/":
                return left / right
            elif expr.operator == "^":
                return left ** right
            else:
                raise ValueError(f"Unknown operator: {expr.operator}")
                
        elif isinstance(expr, UnaryOpExpr):
            operand = self.ast_to_sympy(expr.operand)
            if expr.operator == "-":
                return -operand
            elif expr.operator == "+":
                return operand
            else:
                raise ValueError(f"Unknown unary operator: {expr.operator}")
                
        elif isinstance(expr, DerivativeExpr):
            inner = self.ast_to_sympy(expr.expr)
            var = self.get_symbol(expr.var)
            
            if expr.partial:
                return sp.diff(inner, var, expr.order)
            else:
                # Time derivative
                if expr.var == "t":
                    return sp.diff(inner, self.time_symbol, expr.order)
                else:
                    return sp.diff(inner, var, expr.order)
                    
        elif isinstance(expr, FunctionCallExpr):
            args = [self.ast_to_sympy(arg) for arg in expr.args]
            
            # Built-in functions
            if expr.name == "sin":
                return sp.sin(args[0])
            elif expr.name == "cos":
                return sp.cos(args[0])
            elif expr.name == "tan":
                return sp.tan(args[0])
            elif expr.name == "exp":
                return sp.exp(args[0])
            elif expr.name == "log":
                return sp.log(args[0])
            elif expr.name == "sqrt":
                return sp.sqrt(args[0])
            elif expr.name == "dot":
                # Vector dot product
                return sum(args[i] * args[i+len(args)//2] for i in range(len(args)//2))
            else:
                # Custom function
                func = self.get_function(expr.name)
                return func(*args)
                
        elif isinstance(expr, VectorExpr):
            # Convert to list of sympy expressions
            return [self.ast_to_sympy(comp) for comp in expr.components]
            
        elif isinstance(expr, VectorOpExpr):
            if expr.operation == "grad":
                # Gradient operation
                if expr.left:
                    inner = self.ast_to_sympy(expr.left)
                    # Return gradient as list
                    return [sp.diff(inner, var) for var in ['x', 'y', 'z']]
                else:
                    return sp.symbols('nabla')  # Symbolic gradient
            elif expr.operation == "dot":
                left_vec = self.ast_to_sympy(expr.left)
                right_vec = self.ast_to_sympy(expr.right)
                return sum(a * b for a, b in zip(left_vec, right_vec))
            elif expr.operation == "cross":
                # Cross product
                left_vec = self.ast_to_sympy(expr.left)
                right_vec = self.ast_to_sympy(expr.right)
                if len(left_vec) == 3 and len(right_vec) == 3:
                    return [
                        left_vec[1] * right_vec[2] - left_vec[2] * right_vec[1],
                        left_vec[2] * right_vec[0] - left_vec[0] * right_vec[2], 
                        left_vec[0] * right_vec[1] - left_vec[1] * right_vec[0]
                    ]
                    
        else:
            raise ValueError(f"Cannot convert {type(expr)} to SymPy")

    def derive_equations_of_motion(self, lagrangian: sp.Expr, coordinates: List[str]) -> List[sp.Expr]:
        """Derive Euler-Lagrange equations from Lagrangian"""
        equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot_sym = self.get_symbol(f"{q}_dot")
            
            # Substitute time derivatives
            L_substituted = lagrangian.subs(sp.Derivative(q_sym, self.time_symbol), q_dot_sym)
            
            # ∂L/∂q̇
            dL_dq_dot = sp.diff(L_substituted, q_dot_sym)
            
            # d/dt(∂L/∂q̇) - need to substitute back and differentiate
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)
            
            # ∂L/∂q  
            dL_dq = sp.diff(L_substituted, q_sym)
            
            # Euler-Lagrange equation: d/dt(∂L/∂q̇) - ∂L/∂q = 0
            equation = d_dt_dL_dq_dot - dL_dq
            equations.append(equation)
            
        return equations

    def solve_for_accelerations(self, equations: List[sp.Expr], coordinates: List[str]) -> Dict[str, sp.Expr]:
        """Solve equations of motion for accelerations"""
        accelerations = {}
        
        # Create acceleration symbols
        accel_symbols = [self.get_symbol(f"{q}_ddot") for q in coordinates]
        
        # Solve the system
        try:
            solutions = sp.solve(equations, accel_symbols)
            
            for i, q in enumerate(coordinates):
                accel_key = f"{q}_ddot"
                if accel_key in solutions:
                    accelerations[accel_key] = solutions[accel_key]
                else:
                    # If direct solving fails, try individual equations
                    accelerations[accel_key] = sp.solve(equations[i], accel_symbols[i])[0]
                    
        except Exception as e:
            print(f"Warning: Could not solve equations symbolically: {e}")
            # Return unsolved equations
            for i, q in enumerate(coordinates):
                accelerations[f"{q}_ddot"] = equations[i]
                
        return accelerations

# ===========================
# NUMERICAL SIMULATION ENGINE  
# ===========================

class NumericalSimulator:
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic = symbolic_engine
        self.equations = {}
        self.parameters = {}
        self.initial_conditions = {}

    def set_parameters(self, params: Dict[str, float]):
        """Set physical parameters"""
        self.parameters.update(params)

    def set_initial_conditions(self, conditions: Dict[str, float]):
        """Set initial conditions"""
        self.initial_conditions.update(conditions)

    def compile_equations(self, accelerations: Dict[str, sp.Expr], coordinates: List[str]):
        """Compile symbolic equations to numerical functions"""
        
        # Create state vector: [q1, q1_dot, q2, q2_dot, ...]
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
            
        # Create parameter substitutions
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        # Compile each acceleration equation
        compiled_equations = {}
        
        for q in coordinates:
            accel_key = f"{q}_ddot"
            if accel_key in accelerations:
                # Substitute parameters
                eq = accelerations[accel_key].subs(param_subs)
                
                # Convert to numerical function
                state_symbols = [self.symbolic.get_symbol(var) for var in state_vars]
                compiled_equations[accel_key] = sp.lambdify(state_symbols, eq, 'numpy')
                
        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE system for numerical integration"""
        
        # y = [q1, q1_dot, q2, q2_dot, ...]
        dydt = np.zeros_like(y)
        
        # First half: derivatives of positions are velocities
        for i in range(0, len(y), 2):
            dydt[i] = y[i + 1]  # q_dot
            
        # Second half: derivatives of velocities are accelerations  
        for i, q in enumerate(self.coordinates):
            accel_key = f"{q}_ddot"
            if accel_key in self.equations:
                try:
                    # Call compiled function with current state
                    dydt[2*i + 1] = self.equations[accel_key](*y)
                except Exception as e:
                    print(f"Error evaluating {accel_key}: {e}")
                    dydt[2*i + 1] = 0  # Default to zero acceleration
                    
        return dydt

    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000) -> dict:
        """Run numerical simulation"""
        
        # Set up initial conditions vector
        y0 = []
        for q in self.coordinates:
            y0.append(self.initial_conditions.get(q, 0.0))
            y0.append(self.initial_conditions.get(f"{q}_dot", 0.0))
            
        y0 = np.array(y0)
        
        # Time points
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        # Solve ODE
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span, 
                y0,
                t_eval=t_eval,
                method='DOP853',  # High-accuracy method
                rtol=1e-8,
                atol=1e-10
            )
            
            return {
                'success': solution.success,
                't': solution.t,
                'y': solution.y,
                'coordinates': self.coordinates,
                'state_vars': self.state_vars
            }
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            return {'success': False, 'error': str(e)}

# ===========================
# 3D VISUALIZATION ENGINE
# ===========================

class MechanicsVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.animation = None

    def setup_3d_plot(self, title: str = "Classical Mechanics Simulation"):
        """Setup 3D plotting environment"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title(title, fontsize=16)
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12) 
        self.ax.set_zlabel('Z (m)', fontsize=12)

    def animate_pendulum(self, solution: dict, parameters: dict, system_name: str = "pendulum"):
        """Create animated pendulum visualization"""
        
        if not solution['success']:
            print("Cannot animate failed simulation")
            return None
            
        self.setup_3d_plot(f"{system_name.title()} Animation")
        
        t = solution['t']
        y = solution['y']
        coordinates = solution['coordinates']
        
        # Extract trajectory data based on system type
        if system_name == "pendulum":
            theta = y[0]  # First coordinate
            l = parameters.get('l', 1.0)
            
            # Convert to Cartesian coordinates
            x = l * np.sin(theta)
            y_pos = -l * np.cos(theta)
            z = np.zeros_like(x)
            
            # Set axis limits
            self.ax.set_xlim(-l*1.2, l*1.2)
            self.ax.set_ylim(-l*1.2, l*0.2)
            self.ax.set_zlim(-0.1, 0.1)
            
            # Initialize plot elements
            line, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='red')
            trail, = self.ax.plot([], [], [], '-', alpha=0.3, color='blue')
            
            def animate_frame(frame):
                if frame < len(t):
                    # Pendulum rod and bob
                    line.set_data([0, x[frame]], [0, y_pos[frame]])
                    line.set_3d_properties([0, z[frame]])
                    
                    # Trail of bob
                    trail_length = min(frame, 100)  # Show last 100 points
                    if trail_length > 0:
                        trail.set_data(x[frame-trail_length:frame+1], 
                                     y_pos[frame-trail_length:frame+1])
                        trail.set_3d_properties(z[frame-trail_length:frame+1])
                        
                return line, trail
                
        elif system_name == "double_pendulum":
            theta1 = y[0]
            theta2 = y[2]
            l1 = parameters.get('l1', 1.0)
            l2 = parameters.get('l2', 1.0)
            
            # Positions
            x1 = l1 * np.sin(theta1)
            y1 = -l1 * np.cos(theta1)
            x2 = x1 + l2 * np.sin(theta2)
            y2 = y1 - l2 * np.cos(theta2)
            
            # Set axis limits
            max_reach = l1 + l2
            self.ax.set_xlim(-max_reach*1.1, max_reach*1.1)
            self.ax.set_ylim(-max_reach*1.1, max_reach*0.2)
            self.ax.set_zlim(-0.1, 0.1)
            
            # Plot elements
            line1, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='red')
            line2, = self.ax.plot([], [], [], 'o-', linewidth=3, markersize=8, color='blue')
            trail1, = self.ax.plot([], [], [], '-', alpha=0.3, color='red')
            trail2, = self.ax.plot([], [], [], '-', alpha=0.3, color='blue')
            
            def animate_frame(frame):
                if frame < len(t):
                    # First pendulum
                    line1.set_data([0, x1[frame]], [0, y1[frame]]) 
                    line1.set_3d_properties([0, 0])
                    
                    # Second pendulum
                    line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
                    line2.set_3d_properties([0, 0])
                    
                    # Trails
                    trail_length = min(frame, 200)
                    if trail_length > 0:
                        trail1.set_data(x1[frame-trail_length:frame+1], 
                                      y1[frame-trail_length:frame+1])
                        trail1.set_3d_properties(np.zeros(trail_length+1))
                        
                        trail2.set_data(x2[frame-trail_length:frame+1], 
                                      y2[frame-trail_length:frame+1])
                        trail2.set_3d_properties(np.zeros(trail_length+1))
                        
                return line1, line2, trail1, trail2
                
        else:
            print(f"Animation for {system_name} not implemented yet")
            return None
            
        # Create animation
        interval = max(1, int(1000 * (t[-1] - t[0]) / len(t)))  # Match real time roughly
        self.animation = animation.FuncAnimation(
            self.fig, animate_frame, frames=len(t),
            interval=interval, blit=False, repeat=True
        )
        
        return self.animation

    def plot_energy(self, solution: dict, parameters: dict, system_name: str):
        """Plot energy conservation"""
        
        if not solution['success']:
            return
            
        t = solution['t']
        y = solution['y']
        
        # Calculate energies based on system type
        if system_name == "pendulum":
            theta = y[0]
            theta_dot = y[1]
            
            m = parameters.get('m', 1.0)
            l = parameters.get('l', 1.0)
            g = parameters.get('g', 9.81)
            
            # Kinetic energy
            KE = 0.5 * m * l**2 * theta_dot**2
            
            # Potential energy (taking lowest point as reference)
            PE = m * g * l * (1 - np.cos(theta))
            
            # Total energy
            E_total = KE + PE
            
        elif system_name == "double_pendulum":
            theta1, theta1_dot, theta2, theta2_dot = y[0], y[1], y[2], y[3]
            
            m1 = parameters.get('m1', 1.0)
            m2 = parameters.get('m2', 1.0)
            l1 = parameters.get('l1', 1.0)
            l2 = parameters.get('l2', 1.0)
            g = parameters.get('g', 9.81)
            
            # Kinetic energies (complex for double pendulum)
            KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
            KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                              2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
            KE = KE1 + KE2
            
            # Potential energies
            PE1 = -m1 * g * l1 * np.cos(theta1)
            PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
            PE = PE1 + PE2
            
            E_total = KE + PE
        else:
            print(f"Energy calculation for {system_name} not implemented")
            return
            
        # Plot energy
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(t, KE, label='Kinetic Energy', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Kinetic Energy')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(t, PE, label='Potential Energy', color='blue') 
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Potential Energy')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(t, E_total, label='Total Energy', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Total Energy')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(t, KE, 'r-', label='Kinetic', alpha=0.7)
        plt.plot(t, PE, 'b-', label='Potential', alpha=0.7)
        plt.plot(t, E_total, 'g-', label='Total', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title('Energy Overview')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space (position vs momentum)"""
        
        if not solution['success']:
            return
            
        y = solution['y']
        
        # Extract position and velocity for specified coordinate
        position = y[2 * coordinate_index]      # q
        velocity = y[2 * coordinate_index + 1]  # q_dot
        
        plt.figure(figsize=(10, 8))
        plt.plot(position, velocity, 'b-', alpha=0.7, linewidth=1)
        plt.plot(position[0], velocity[0], 'go', markersize=8, label='Start')
        plt.plot(position[-1], velocity[-1], 'ro', markersize=8, label='End')
        
        plt.xlabel(f'Position {solution["coordinates"][coordinate_index]}')
        plt.ylabel(f'Velocity {solution["coordinates"][coordinate_index]}_dot')
        plt.title('Phase Space Trajectory')
        plt.grid(True)
        plt.legend()
        plt.show()

# ===========================
# COMPLETE PHYSICS COMPILER
# ===========================

class PhysicsCompiler:
    def __init__(self):
        self.ast = []
        self.variables = {}
        self.definitions = {}
        self.system_name = "unnamed_system"
        self.lagrangian = None
        self.transforms = {}
        self.initial_conditions = {}
        
        # Engines
        self.symbolic = SymbolicEngine()
        self.simulator = NumericalSimulator(self.symbolic)
        self.visualizer = MechanicsVisualizer()

    def compile_dsl(self, dsl_source: str) -> dict:
        """Complete compilation pipeline"""
        
        print("🚀 Starting Physics Compilation...")
        
        try:
            # Tokenization
            print("📝 Tokenizing...")
            tokens = tokenize(dsl_source)
            print(f"   Found {len(tokens)} tokens")
            
            # Parsing  
            print("🔍 Parsing AST...")
            parser = MechanicsParser(tokens)
            self.ast = parser.parse()
            print(f"   Generated {len(self.ast)} AST nodes")
            
            # Semantic analysis
            print("🧠 Analyzing semantics...")
            self.analyze_semantics()
            
            # Generate equations
            print("⚡ Deriving equations of motion...")
            equations = self.derive_equations()
            
            # Prepare simulation
            print("🔧 Setting up simulation...")
            self.setup_simulation(equations)
            
            print("✅ Compilation successful!")
            
            return {
                'success': True,
                'system_name': self.system_name,
                'coordinates': list(self.get_coordinates()),
                'equations': equations,
                'simulator': self.simulator
            }
            
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def analyze_semantics(self):
        """Analyze AST and extract system information"""
        
        for node in self.ast:
            if isinstance(node, SystemDef):
                self.system_name = node.name
                
            elif isinstance(node, VarDef):
                self.variables[node.name] = {
                    'type': node.vartype,
                    'unit': node.unit,
                    'vector': node.vector
                }
                
            elif isinstance(node, DefineDef):
                self.definitions[node.name] = {
                    'args': node.args,
                    'body': node.body
                }
                
            elif isinstance(node, LagrangianDef):
                self.lagrangian = node.expr
                
            elif isinstance(node, TransformDef):
                self.transforms[node.var] = {
                    'type': node.coord_type,
                    'expression': node.expr
                }
                
            elif isinstance(node, InitialCondition):
                self.initial_conditions.update(node.conditions)

    def get_coordinates(self) -> List[str]:
        """Extract generalized coordinates from system"""
        coordinates = []
        
        for var_name, var_info in self.variables.items():
            # Look for angle, position, or coordinate variables
            if (var_info['type'] in ['Angle', 'Position', 'Coordinate'] or
                var_name in ['theta', 'theta1', 'theta2', 'x', 'y', 'z', 'r', 'phi']):
                coordinates.append(var_name)
                
        return coordinates

    def derive_equations(self) -> Dict[str, sp.Expr]:
        """Derive equations of motion from Lagrangian"""
        
        if not self.lagrangian:
            raise ValueError("No Lagrangian defined in system")
            
        # Convert Lagrangian to SymPy
        L_sympy = self.symbolic.ast_to_sympy(self.lagrangian)
        
        # Get coordinates
        coordinates = self.get_coordinates()
        if not coordinates:
            raise ValueError("No generalized coordinates found")
            
        print(f"   Coordinates: {coordinates}")
        print(f"   Lagrangian: {L_sympy}")
        
        # Derive Euler-Lagrange equations
        eq_list = self.symbolic.derive_equations_of_motion(L_sympy, coordinates)
        
        # Solve for accelerations
        accelerations = self.symbolic.solve_for_accelerations(eq_list, coordinates)
        
        return accelerations

    def setup_simulation(self, equations: Dict[str, sp.Expr]):
        """Setup numerical simulator"""
        
        # Extract parameters from variables
        parameters = {}
        for var_name, var_info in self.variables.items():
            if var_info['type'] in ['Real', 'Mass', 'Length', 'Acceleration']:
                # Use default values or ask user to provide them
                if var_name == 'g':
                    parameters[var_name] = 9.81
                elif var_name in ['m', 'm1', 'm2']:
                    parameters[var_name] = 1.0
                elif var_name in ['l', 'l1', 'l2']:
                    parameters[var_name] = 1.0
                else:
                    parameters[var_name] = 1.0  # Default value
                    
        self.simulator.set_parameters(parameters)
        self.simulator.set_initial_conditions(self.initial_conditions)
        
        # Compile equations
        coordinates = self.get_coordinates()
        self.simulator.compile_equations(equations, coordinates)

    def simulate(self, t_span: Tuple[float, float] = (0, 10), num_points: int = 1000):
        """Run simulation"""
        return self.simulator.simulate(t_span, num_points)

    def animate(self, solution: dict):
        """Create animation"""
        parameters = self.simulator.parameters
        return self.visualizer.animate_pendulum(solution, parameters, self.system_name)

    def plot_energy(self, solution: dict):
        """Plot energy analysis"""
        parameters = self.simulator.parameters
        self.visualizer.plot_energy(solution, parameters, self.system_name)

    def plot_phase_space(self, solution: dict, coordinate_index: int = 0):
        """Plot phase space"""
        self.visualizer.plot_phase_space(solution, coordinate_index)

# ===========================
# EXAMPLE SYSTEMS
# ===========================

# Simple Pendulum
SIMPLE_PENDULUM_DSL = """
\\system{simple_pendulum}

\\defvar{theta}{Angle}{rad}
\\defvar{m}{Mass}{kg}
\\defvar{l}{Length}{m}
\\defvar{g}{Acceleration}{m/s^2}

\\define{\\op{kinetic}(m, l, theta_dot) = 0.5 * m * l^2 * theta_dot^2}
\\define{\\op{potential}(m, g, l, theta) = m * g * l * (1 - \\cos{theta})}

\\lagrangian{kinetic(m, l, \\dot{theta}) - potential(m, g, l, theta)}

\\initial{theta=0.5, theta_dot=0}

\\solve{euler_lagrange}
\\animate{pendulum}
"""
