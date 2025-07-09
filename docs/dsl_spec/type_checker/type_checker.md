Summary:
This module provides semantic analysis through a type checker that validates:

Variable definitions exist before use

Expressions are dimensionally consistent

Units combine logically (e.g., no length + time)

It uses the unit system defined in physical_units_system.py to enforce these rules.

Design Rationale:
Guarantees physical and logical soundness of DSL programs.

Supports partial checking (warnings only) to allow exploratory definitions.

Features:
First pass: collects all VarDefs into a symbol table.

Second pass: checks unit compatibility in expressions and function definitions.

Detects undefined variables in operators, boundaries, and symmetries.
