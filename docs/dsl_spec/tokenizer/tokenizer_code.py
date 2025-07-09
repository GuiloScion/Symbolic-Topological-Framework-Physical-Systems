def tokenize(source: str) -> List[Token]:
    tokens = []
    for match in token_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        position = match.start()
        if kind != "WHITESPACE":
            tokens.append(Token(kind, value, position))
    return tokens
