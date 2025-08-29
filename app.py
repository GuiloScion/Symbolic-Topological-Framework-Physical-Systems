import streamlit as st  
from complete_physics_dsl import *  # Import everything from your backend module  
  
# Initialize the Streamlit app  
st.title("Physics DSL Compiler")  
st.write("This application compiles your physics DSL code and simulates the results.")  
  
# Text area for DSL input  
dsl_input = st.text_area("Enter your DSL code here:", height=200)  
  
# Compile button  
if st.button("Compile"):  
    # Step 1: Tokenization  
    st.write("📝 Tokenizing...")  
    tokens = tokenize(dsl_input)  # Call your tokenization function  
    st.write(f"Found {len(tokens)} tokens:")  
    for token in tokens:  
        st.write(token)  
  
    # Step 2: Parsing  
    st.write("🔍 Parsing the tokens...")  
    try:  
        # Create an instance of MechanicsParser  
        parser = MechanicsParser(tokens)  
        ast = parser.parse()  # Call the parse method on the instance  
        st.success("Parsing Successful!")  
  
        # Step 3: Displaying the AST  
        st.write("### Generated AST Nodes:")  
        for node in ast:  
            st.write(node)  
  
        # Step 4: Extract Coordinates  
        coordinates = []  
        for node in ast:  
            if isinstance(node, VarDef) and node.vartype in ['Angle', 'Position', 'Coordinate']:  
                coordinates.append(node.name)  
  
        if not coordinates:  
            st.error("No valid coordinates found in the DSL.")  
        else:  
            st.write("### Extracted Coordinates:")  
            st.write(coordinates)  
  
            # Step 5: Deriving Equations  
            st.write("⚡ Deriving Equations of Motion...")  
            symbolic_engine = SymbolicEngine()  # Instantiate SymbolicEngine  
            equations = symbolic_engine.derive_equations_of_motion(ast, coordinates)  # Call the method  
            st.write("### Equations of Motion:")  
            st.write(equations)  
  
            # Step 6: Running Simulation (if applicable)  
            st.write("🔧 Running Simulation...")  
            solution = run_simulation(equations)  # Assuming you have a function to run simulations  
            if solution['success']:  
                st.write("### Simulation Results:")  
                st.line_chart(solution['y'])  # Plot simulation results  
                st.write("Time:", solution['t'])  
            else:  
                st.error("Simulation failed!")  
  
    except Exception as e:  
        st.error(f"Error during parsing or simulation: {e}")  
