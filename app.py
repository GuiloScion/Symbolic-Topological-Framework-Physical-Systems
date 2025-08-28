import streamlit as st  
from your_compiler_file import PhysicsCompiler  # Adjust to your actual file name  
  
# Initialize the Streamlit app  
st.title("Physics Compiler")  
st.write("This application compiles your physics DSL code and simulates the results.")  
  
# Text area for DSL input  
dsl_input = st.text_area("Enter your DSL code here:", height=200)  
  
# Compile button  
if st.button("Compile"):  
    compiler = PhysicsCompiler()  
    result = compiler.compile_dsl(dsl_input)  
  
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
            solution = result['simulator'].simulate((0, 10))  # Example time span  
            st.write("### Simulation Results:")  
            if solution['success']:  
                st.line_chart(solution['y'])  # Plot simulation results  
            else:  
                st.error("Simulation failed!")  
  
    else:  
        st.error("Compilation Failed!")  
        st.write("Error:", result['error'])  
