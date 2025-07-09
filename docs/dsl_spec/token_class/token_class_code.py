@dataclass
class Token:
    type: str
    value: str
    position: int = 0  # Position in source for better error reporting

    def __repr__(self):
        return f"{self.type}:{self.value}@{self.position}"
