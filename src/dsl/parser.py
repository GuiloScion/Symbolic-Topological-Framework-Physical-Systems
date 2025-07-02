class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def match(self, expected_type: str) -> Optional[Token]:
        token = self.peek()
        if token and token.type == expected_type:
            self.pos += 1
            return token
        return None

    def expect(self, expected_type: str) -> Token:
        token = self.match(expected_type)
        if not token:
            current = self.peek()
            if current:
                raise SyntaxError(f"Expected {expected_type} but got {current.type} '{current.value}' at position {current.position}")
            else:
                raise SyntaxError(f"Expected {expected_type} but reached end of input")
        return token

    def parse(self) -> List[ASTNode]:
        nodes = []
        while self.pos < len(self.tokens):
            token = self.peek()
            if token.type == "COMMAND":
                if token.value == r"\defvar":
                    nodes.append(self.parse_defvar())
                elif token.value == r"\define":
                    nodes.append(self.parse_define())
                elif token.value == r"\boundary":
                    nodes.append(self.parse_boundary())
                elif token.value == r"\symmetry":
                    nodes.append(self.parse_symmetry())
                else:
                    raise SyntaxError(f"Unknown command: {token.value} at position {token.position}")
            else:
                self.pos += 1
        return nodes

    def parse_defvar(self) -> VarDef:
        self.expect("COMMAND")  # \defvar
        self.expect("LBRACE")
        name = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        vartype = self.expect("IDENT").value
        self.expect("RBRACE")
        self.expect("LBRACE")
        # Unit can be either an identifier or a number (for dimensionless quantities)
        unit_token = self.peek()
        if unit_token and unit_token.type == "IDENT":
            unit = self.expect("IDENT").value
        elif unit_token and unit_token.type == "NUMBER":
            unit = self.expect("NUMBER").value
        else:
            raise SyntaxError(f"Expected unit (IDENT or NUMBER) but got {unit_token.type if unit_token else 'EOF'}")
        self.expect("RBRACE")
        return VarDef(name, vartype, unit)

    def parse_define(self) -> Define:
        self.expect("COMMAND")  # \define
        self.expect("LBRACE")

        self.expect("COMMAND")  # \op
        self.expect("LBRACE")
        op_name = self.expect("IDENT").value
        self.expect("RBRACE")

        self.expect("LPAREN")
        args = []
        args.append(self.expect("IDENT").value)
        while self.match("COMMA"):
            args.append(self.expect("IDENT").value)
        self.expect("RPAREN")

        self.expect("EQUALS")

        # Parse the right-hand side expression
        rhs = self.parse_expression()

        self.expect("RBRACE")

        return Define(Op(op_name, args), rhs)

    def parse_expression(self) -> Expression:
        """Parse expressions with operator precedence"""
        return self.parse_additive()

    def parse_additive(self) -> Expression:
        """Parse + and - operators (lowest precedence)"""
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
        """Parse * and / operators (medium precedence)"""
        left = self.parse_power()

        while True:
            if self.match("MULTIPLY"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "*", right)
            elif self.match("DIVIDE"):
                right = self.parse_power()
                left = BinaryOpExpr(left, "/", right)
            else:
                break

        return left

    def parse_power(self) -> Expression:
        """Parse ^ operator (highest precedence)"""
        left = self.parse_primary()

        if self.match("POWER"):
            right = self.parse_power()  # Right associative
            return BinaryOpExpr(left, "^", right)

        return left

    def parse_primary(self) -> Expression:
        """Parse primary expressions (numbers, identifiers, parentheses)"""
        token = self.peek()

        if self.match("NUMBER"):
            return NumberExpr(float(self.tokens[self.pos - 1].value))

        if self.match("IDENT"):
            return IdentExpr(self.tokens[self.pos - 1].value)

        if self.match("LPAREN"):
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        if self.match("MINUS"):
            operand = self.parse_primary()
            return UnaryOpExpr("-", operand)

        if token:
            raise SyntaxError(f"Unexpected token {token.type} '{token.value}' at position {token.position}")
        else:
            raise SyntaxError("Unexpected end of input")

    def parse_boundary(self) -> Boundary:
        self.expect("COMMAND")  # \boundary
        self.expect("LBRACE")
        expr = self.expect("IDENT").value
        self.expect("RBRACE")
        return Boundary(expr)

    def parse_symmetry(self) -> Symmetry:
        self.expect("COMMAND")  # \symmetry
        self.expect("LBRACE")
        law = self.expect("IDENT").value
        self.expect("COMMAND")  # \invariant
        invariant = self.expect("IDENT").value
        self.expect("RBRACE")
        return Symmetry(law, invariant)