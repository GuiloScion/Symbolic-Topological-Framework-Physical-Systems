import streamlit as st
from complete_physics_dsl import *

st.title("Physics DSL Compiler (Debug Mode)")

dsl_input = st.text_area("Enter your DSL code here:", height=200, value="""\\system{simple_pendulum} \\defvar{theta}{Angle}{rad} \\defvar{m}{Mass}{kg} \\defvar{l}{Length}{m} \\defvar{g}{Acceleration}{m/s^2} \\define{\\op{kinetic}(m, l, theta_dot) = 0.5 * m * l^2 * theta_dot^2} \\define{\\op{potential}(m, g, l, theta) = m * g * l * (1 - \\cos{theta})} \\lagrangian{kinetic(m, l, \\dot{theta}) - potential(m, g, l, theta)} \\initial{theta=0.5, theta_dot=0} \\solve{euler_lagrange} \\animate{pendulum}""")

if st.button("Compile & Debug"):
    # Step 1: Tokenization
    st.write("📝 Tokenizing...")
    tokens = tokenize(dsl_input)
    st.write(f"Tokens ({len(tokens)}):")
    for t in tokens:
        st.write(repr(t))

    # Step 2: Parsing
    st.write("🔍 Parsing the tokens...")
    parser = MechanicsParser(tokens)
    ast = parser.parse()
    st.write("AST node count:", len(ast))
    st.write("Raw AST value:", ast)
    st.write("### AST Node Dumps:")
    for node in ast:
        st.json({"type": type(node).__name__, **vars(node)})

    # Step 3: Coordinates extraction (case-insensitive, substring match)
    coordinates = []
    COORD_SUBSTRINGS = ["angle", "position", "coordinate"]
    for node in ast:
        if isinstance(node, VarDef):
            vartype = str(getattr(node, "vartype", "")).strip().lower()
            if any(sub in vartype for sub in COORD_SUBSTRINGS):
                coordinates.append(node.name)
    st.write("Extracted coordinates:", coordinates)
