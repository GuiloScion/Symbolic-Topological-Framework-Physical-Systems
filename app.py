import streamlit as st  
from physics_compiler import *  # Import everything from your backend module  
  
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
  
        # Step 4: Deriving Equations (if applicable)  
        st.write("⚡ Deriving Equations of Motion...")  
        equations = derive_equations(ast)  # Assuming you have a function for this  
        st.write("### Equations of Motion:")  
        st.write(equations)  
  
        # Step 5: Running Simulation (if applicable)  
        st.write("🔧 Running Simulation...")  
        solution = run_simulation(equations)  # Assuming you have a function to run simulations  
        if solution['success']:  
            st.write("### Simulation Results:")  
            st.line_chart(solution['y'])  # Plot simulation results  
            st.write("Time:", solution['t'])  
            st.write("State Variables:", solution['state_vars'])  
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
