#(Excerpt; Complete in source(src/legacy))

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
