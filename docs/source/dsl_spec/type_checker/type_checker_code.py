#(Excerpt below; complete in source)

class TypeChecker:
    def __init__(self):
        self.variables = {}  # name -> (type, Unit)
        self.operators = {}  # name -> signature

    def get_unit(self, unit_str): ...
    def check_expression_type(self, expr: Expression): ...
    def check(self, ast: List[ASTNode]):
