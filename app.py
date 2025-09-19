import streamlit as st  
from complete_physics_dsl import *  # Import everything from your backend module  

st.set_page_config(page_title="Physics DSL Compiler", layout="wide")
st.title("Physics DSL Compiler")
st.write("Compile your DSL, view the AST, extract coordinates, derive equations, and simulate results.")

dsl_input = st.text_area("Enter your DSL code here:", height=200)  

if st.button("Compile & Simulate"):
    # Step 1: Tokenization
    try:
        st.write("📝 Tokenizing...")
        tokens = tokenize(dsl_input)
        st.write(f"Found {len(tokens)} tokens:")
        st.json([str(token) for token in tokens])
    except Exception as e:
        st.error(f"Tokenization error: {e}")
        st.stop()

    # Step 2: Parsing
    try:
        st.write("🔍 Parsing the tokens...")
        parser = MechanicsParser(tokens)
        ast = parser.parse()
        st.success("Parsing Successful!")
        st.write("### Generated AST Nodes (Full Attribute Dump):")
        for node in ast:
            st.json({"type": type(node).__name__, **vars(node)})
    except Exception as e:
        st.error(f"Parsing error: {e}")
        st.stop()

    # Step 3: Extract coordinates (robust logic)
    st.write("### Extracting Physical Coordinates")
    coordinates = []
    for node in ast:
        # Check if it's a VarDef
        if isinstance(node, VarDef):
            if (
                hasattr(node, "vartype")
                and str(node.vartype).strip().lower() in ['angle', 'position', 'coordinate']
            ):
                coordinates.append(node.name)
            # Also consider vector coordinates
            elif hasattr(node, "vector") and node.vector:
                coordinates.append(node.name + " (vector)")
    if not coordinates:
        st.error("No valid coordinates found in the DSL. See node details above for troubleshooting.")
        st.info("If you believe this is a bug, check your VarDef nodes in the AST dump above.")
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
