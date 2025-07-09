class Unit:
    def __init__(self, dimensions: Dict[str, int]):
        self.dimensions = dimensions

    def __mul__(self, other):
        result = {}
        all_dims = set(self.dimensions) | set(other.dimensions)
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) + other.dimensions.get(dim, 0)
        return Unit(result)

    def __truediv__(self, other):
        result = {}
        all_dims = set(self.dimensions) | set(other.dimensions)
        for dim in all_dims:
            result[dim] = self.dimensions.get(dim, 0) - other.dimensions.get(dim, 0)
        return Unit(result)

    def __pow__(self, exponent):
        return Unit({dim: exp * exponent for dim, exp in self.dimensions.items()})

    def is_compatible(self, other):
        return self.dimensions == other.dimensions

    def __repr__(self):
        return f"Unit({self.dimensions})"

# Standard units
UNITS = {
    "meter": Unit({"length": 1}),
    "m": Unit({"length": 1}),
    "second": Unit({"time": 1}),
    "s": Unit({"time": 1}),
    "kilogram": Unit({"mass": 1}),
    "kg": Unit({"mass": 1}),
    "kelvin": Unit({"temperature": 1}),
    "K": Unit({"temperature": 1}),
    "dimensionless": Unit({}),
    "1": Unit({}),
    "energy": Unit({"mass": 1, "length": 2, "time": -2}),
    "force": Unit({"mass": 1, "length": 1, "time": -2}),
}
