TOKEN_TYPES = [
    ("COMMAND", r"\\[a-zA-Z]+"),          # LaTeX-style command
    ("LBRACE", r"\{"),                    # Left curly brace
    ("RBRACE", r"\}"),                    # Right curly brace
    ("LPAREN", r"\("),                    # Left parenthesis
    ("RPAREN", r"\)"),                    # Right parenthesis
    ("PLUS", r"\+"),                      # Addition operator
    ("MINUS", r"-"),                      # Subtraction operator
    ("MULTIPLY", r"\*"),                  # Multiplication operator
    ("DIVIDE", r"/"),                     # Division operator
    ("POWER", r"\^"),                     # Exponentiation operator
    ("EQUALS", r"="),                     # Equals sign
    ("COMMA", r","),                      # Comma separator
    ("NUMBER", r"\d+(\.\d+)?"),           # Integer or floating-point number
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"), # Identifier
    ("WHITESPACE", r"\s+"),               # Whitespace
]

# Combine into one regex with named groups
token_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES)
token_pattern = re.compile(token_regex)
