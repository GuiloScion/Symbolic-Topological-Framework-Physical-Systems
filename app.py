import streamlit as st
from complete_physics_dsl import *

st.set_page_config(page_title="Symbolic Topological Physical Systems", layout="wide")
st.title("Symbolic Topological Physical Systems Framework")
st.write("Compile your DSL, view the AST, extract physical coordinates, derive equations, and simulate dynamics.")

dsl_input = st.text_area("Enter your DSL code here:", height=200)

if st.button("Compile & Simulate"):
    # Step 1: Tokenization
    st.write("📝 Tokenizing...")
    try:
        tokens = tokenize(dsl_input)
        st.write(f"Found {len(tokens)} tokens:")
        st.json([str(token) for token in tokens])
    except Exception as e:
        st.error(f"Tokenization error: {e}")
        st.stop()

    # Step 2: Parsing
    st.write("🔍 Parsing the tokens...")
    try:
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        st.success("Parsing Successful!")
        st.write("### AST Nodes")
        for node in ast:
            st.write(repr(node))
    except Exception as e:
        st.error(f"Parsing error: {e}")
        st.stop()

    # Step 3: Extract coordinates/physical quantities (robust)
    st.write("### Extracting Physical Coordinates")
    coordinates = []
    for node in ast:
        # Support both legacy and new VarDef (with vector attribute)
        if (
            hasattr(node, "vartype")
            and hasattr(node, "name")
            and isinstance(node, VarDef)
            and str(node.vartype).strip().lower() in ['angle', 'position', 'coordinate']
        ):
            coordinates.append(node.name)
        # Optionally: add vector support or other types
        elif hasattr(node, "vector") and getattr(node, "vector", False) and hasattr(node, "name"):
            coordinates.append(node.name + " (vector)")
    if not coordinates:
        st.error("No valid coordinates found in the DSL. Try checking your \\defvar definitions.")
        st.info("AST node details:")
        for node in ast:
            st.json(vars(node))
        st.stop()
    else:
        st.success("Coordinates extracted:")
        st.write(coordinates)

    # Step 4: Derive equations
    try:
        st.write("⚡ Deriving Equations of Motion...")
        symbolic_engine = SymbolicEngine()
        equations = symbolic_engine.derive_equations_of_motion(ast, coordinates)
        st.write("### Equations of Motion:")
        st.write(equations)
    except Exception as e:
        st.error(f"Equation derivation error: {e}")
        st.stop()

    # Step 5: Run simulation
    try:
        st.write("🔧 Running Simulation...")
        solution = run_simulation(equations)
        if solution.get('success'):
            st.write("### Simulation Results:")
            st.line_chart(solution['y'])
            st.write("Time:", solution['t'])
        else:
            st.error("Simulation failed!")
            st.json(solution)
    except Exception as e:
        st.error(f"Simulation error: {e}")
