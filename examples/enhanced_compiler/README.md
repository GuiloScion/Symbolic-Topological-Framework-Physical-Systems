# Enhanced DSL Compiler Example

This example (`example_usage.py`) demonstrates the core functionality of the Symbolic-Topological Framework's enhanced compiler. It showcases:

- Definition of variables with types and units.
- Definition of operators with mathematical expressions, including powers and constants.
- The full pipeline: tokenization, parsing, AST generation, type checking (with dimensional analysis), and Intermediate Representation (IR) compilation.
- Specification of boundary conditions and symmetry invariants within the DSL.

## How to Run

To run this example, navigate to the root of your project and execute the Python file:

```bash
python examples/enhanced_compiler/example_usage.py