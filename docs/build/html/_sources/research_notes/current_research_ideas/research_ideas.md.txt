# Current Research Ideas - 2025-7-9

## Idea 1: Enhancing Sheaf-based Encodings for Dynamic Systems
* **Problem:** Current sheaf encodings are strong for static spatial relationships, but dynamic evolution is challenging.
* **Proposed Solution:** Investigate spatio-temporal sheaves or integrate concepts from flow categories. Explore how persistent homology could track changes over time in a system's topological features.
* **Initial thoughts:** Could we define a category of "evolving sheaves" with time-dependent morphisms?

## Idea 2: Integrating Formal Verification with PDE Solvers
* **Problem:** Symbolic PDE solvers can be complex, making correctness hard to ensure.
* **Proposed Solution:** Use formal methods (e.g., theorem provers like Lean or Coq) to verify the correctness of symbolic PDE transformations and solution steps.
* **Initial thoughts:** Start with simple cases like linear first-order PDEs. How to bridge the gap between symbolic Python expressions and formal proof languages?

## Idea 3: Neural-Symbolic Hybrid for Invariant Discovery
* **Problem:** Manually discovering invariants is tedious; purely neural methods lack interpretability.
* **Proposed Solution:** Develop a neural-symbolic approach where neural networks propose potential invariants, which are then formally checked and refined by symbolic reasoning modules.
* **Initial thoughts:** Use physics-informed neural networks (PINNs) to learn potential conservation laws from data, then convert NN outputs to symbolic expressions for verification.

---
*(Add date to filename and update content as research progresses)*
