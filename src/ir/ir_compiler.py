class CategoricalIR:
    def __init__(self):
        self.objects: Dict[str, Dict] = {}
        self.morphisms: List[Tuple[str, str, str, str, Dict]] = []
        self.functors: List[Dict] = []

    def add_object(self, name: str, properties: Dict):
        """Add an object to the category"""
        self.objects[name] = properties

    def add_morphism(self, name: str, domain: str, codomain: str, description: str, properties: Dict = None):
        """Add a morphism to the category"""
        if properties is None:
            properties = {}
        self.morphisms.append((name, domain, codomain, description, properties))

    def add_functor(self, name: str, source_category: str, target_category: str, object_map: Dict, morphism_map: Dict):
        """Add a functor between categories"""
        self.functors.append({
            "name": name,
            "source": source_category,
            "target": target_category,
            "object_map": object_map,
            "morphism_map": morphism_map
        })

    def compose_morphisms(self, f_name: str, g_name: str) -> Optional[str]:
        """Attempt to compose two morphisms f;g"""
        f_morph = next((m for m in self.morphisms if m[0] == f_name), None)
        g_morph = next((m for m in self.morphisms if m[0] == g_name), None)

        if f_morph and g_morph and f_morph[2] == g_morph[1]:  # codomain(f) == domain(g)
            comp_name = f"{f_name}_{g_name}"
            comp_desc = f"Composition of {f_morph[3]} and {g_morph[3]}"
            self.add_morphism(comp_name, f_morph[1], g_morph[2], comp_desc)
            return comp_name
        return None

class IRCompiler:
    def __init__(self):
        self.ir = CategoricalIR()

    def compile(self, ast: List[ASTNode]):
        print("\n[Enhanced IR] Compiling to Enhanced Categorical IR...")

        # Process variable definitions as objects
        for node in ast:
            if isinstance(node, VarDef):
                self.ir.add_object(node.name, {
                    "type": node.vartype,
                    "unit": node.unit,
                    "category": "PhysicalQuantity"
                })

        # Process definitions as morphisms
        for node in ast:
            if isinstance(node, Define):
                if isinstance(node.lhs, Op):
                    morphism_name = f"define_{node.lhs.name}"
                    domain = node.lhs.args[0] if node.lhs.args else "Unknown"
                    codomain = domain  # Could be refined based on return type analysis
                    law_desc = f"{node.lhs.name}({', '.join(node.lhs.args)}) = {self.expr_to_string(node.rhs)}"

                    self.ir.add_morphism(morphism_name, domain, codomain, law_desc, {
                        "type": "PhysicalLaw",
                        "operator": node.lhs.name,
                        "arity": len(node.lhs.args)
                    })

            elif isinstance(node, Boundary):
                var = node.expr
                self.ir.add_morphism(f"boundary_{var}", var, "BoundarySpace", "Boundary condition", {
                    "type": "BoundaryCondition"
                })

            elif isinstance(node, Symmetry):
                self.ir.add_morphism(f"symmetry_{node.law}", "SymmetryGroup", "SymmetryGroup",
                                   f"Invariant under {node.invariant}", {
                    "type": "Symmetry",
                    "law": node.law,
                    "invariant": node.invariant
                })

        # Add identity morphisms for all objects
        for obj_name in self.ir.objects:
            self.ir.add_morphism(f"id_{obj_name}", obj_name, obj_name, f"Identity on {obj_name}", {
                "type": "Identity"
            })

        self.pretty_print()

    def expr_to_string(self, expr: Expression) -> str:
        """Convert expression AST back to string representation"""
        if isinstance(expr, NumberExpr):
            return str(expr.value)
        elif isinstance(expr, IdentExpr):
            return expr.name
        elif isinstance(expr, BinaryOpExpr):
            left_str = self.expr_to_string(expr.left)
            right_str = self.expr_to_string(expr.right)
            return f"({left_str} {expr.operator} {right_str})"
        elif isinstance(expr, UnaryOpExpr):
            operand_str = self.expr_to_string(expr.operand)
            return f"{expr.operator}{operand_str}"
        return str(expr)

    def pretty_print(self):
        print("\n[Enhanced IR] Objects:")
        for obj, props in self.ir.objects.items():
            print(f" - {obj} : {props['type']} [{props['unit']}] in {props['category']}")

        print("\n[Enhanced IR] Morphisms:")
        for name, domain, codomain, desc, props in self.ir.morphisms:
            morph_type = props.get('type', 'Unknown')
            print(f" - {name}: {domain} -> {codomain} | {desc} [{morph_type}]")

        print("\n[Enhanced IR] Categorical Properties:")
        print(f" - Objects: {len(self.ir.objects)}")
        print(f" - Morphisms: {len(self.ir.morphisms)}")
        print(f" - Functors: {len(self.ir.functors)}")