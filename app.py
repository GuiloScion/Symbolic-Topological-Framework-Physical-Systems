import streamlit as st  
from complete_physics_dsl import *  # Import everything from the backend  
  
# Initialize the Streamlit app  
st.title("Physics DSL Compiler")  
st.write("This application compiles your physics DSL code and simulates the results.")  
  
# Text area for DSL input  
dsl_input = st.text_area("Enter your DSL code here:", height=200)  
  
# Compile button  
if st.button("Compile"):  
    # Step 1: Tokenization  
    st.write("📝 Tokenizing...")  
    tokens = tokenize(dsl_input)  # Call the tokenize function  
    st.write(f"Found {len(tokens)} tokens:")  
    for token in tokens:  
        st.write(token)  
  
    # Step 2: Parsing  
    st.write("🔍 Parsing the tokens...")  
    try:  
        ast = parse(tokens)  # Call the parse function  
  
        # Step 3: Displaying the AST  
        st.success("Parsing Successful!")  
        st.write("### Generated AST Nodes:")  
        for node in ast:  
            st.write(node)  
  
        # Step 4: Deriving Equations (if applicable)  
        equations = derive_equations(ast)  # Assuming you have a function for this  
        st.write("### Equations of Motion:")  
        st.write(equations)  
  
        # Step 5: Running Simulation (if applicable)  
        solution = run_simulation(equations)  # Assuming you have a function to run simulations  
        if solution['success']:  
            st.write("### Simulation Results:")  
            st.line_chart(solution['y'])  # Plot simulation results  
            st.write("Time:", solution['t'])  
        else:  
            st.error("Simulation failed!")  
  
        # Display any additional results, like energy analysis  
        if 'energy' in solution:  
            st.write("### Energy Analysis:")  
            st.write("Potential Energy:", solution['energy'].get('potential', 'N/A'))  
            st.write("Kinetic Energy:", solution['energy'].get('kinetic', 'N/A'))  
            st.write("Total Energy:", solution['energy'].get('total', 'N/A'))  
  
    except Exception as e:  
        st.error(f"Error during parsing or simulation: {e}")  
