import streamlit as st  
from complete_physics_dsl import *  # Import everything from your backend module  
  
# Initialize the Streamlit app  
st.title("Physics DSL Compiler")  
st.write("This application compiles your physics DSL code and simulates the results.")  
  
# Text area for DSL input  
dsl_input = st.text_area("Enter your DSL code here:", height=200)  
  
# Compile button  
if st.button("Compile"):  
    try:  
        st.write("📝 Tokenizing...")  
        tokens = tokenize(dsl_input)  # Call your tokenization function  
        st.write(f"Found {len(tokens)} tokens:")  
        for token in tokens:  
            st.write(token)  
  
        st.write("🔍 Parsing AST...")  
        result = compile_dsl(dsl_input)  # Call the compile function  
  
        if result['success']:  
            st.success("Compilation Successful!")  
  
            # Display the generated AST  
            st.write("### Generated AST Nodes:")  
            for node in result['ast']:  
                st.write(node)  
  
            # Display equations of motion  
            st.write("### Equations of Motion:")  
            st.write(result['equations'])  
  
            # Run simulation if applicable  
            if 'simulator' in result:  
                st.write("🔧 Running Simulation...")  
                solution = result['simulator'].simulate((0, 10))  # Example time span  
                if solution['success']:  
                    st.line_chart(solution['y'])  # Plot simulation results  
                    st.write("### Simulation Results:")  
                    st.write("Time:", solution['t'])  
                    st.write("State Variables:", solution['state_vars'])  
                else:  
                    st.error("Simulation failed!")  
  
            # Energy analysis  
            st.write("### Energy Analysis:")  
            if 'energy' in result:  
                st.write("Potential Energy:", result['energy']['potential'])  
                st.write("Kinetic Energy:", result['energy']['kinetic'])  
                st.write("Total Energy:", result['energy']['total'])  
  
        else:  
            st.error("Compilation Failed!")  
            st.write("Error:", result['error'])  
  
    except Exception as e:  
        st.error(f"An error occurred: {e}")  
