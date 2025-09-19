import streamlit as st
from complete_physics_dsl import *

st.title("Physics DSL Compiler")

dsl_input = st.text_area("Enter your DSL code here:", height=300, value="""
\\system{simple_pendulum}
\\defvar{theta}{Angle}{rad}
\\defvar{m}{Mass}{kg}
\\defvar{l}{Length}{m}
\\defvar{g}{Acceleration}{m/s^2}
\\define{\\op{kinetic}(m, l, theta_dot) = 0.5 * m * l^2 * theta_dot^2}
\\define{\\op{potential}(m, g, l, theta) = m * g * l * (1 - \\cos{theta})}
\\lagrangian{kinetic(m, l, \\dot{theta}) - potential(m, g, l, theta)}
\\initial{theta=0.5, theta_dot=0}
\\solve{euler_lagrange}
\\animate{pendulum}
""")

if st.button("Compile & Simulate"):
    st.write("🚀 Compiling DSL...")
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_input)
    
    if not result['success']:
        st.error(f"Compilation failed: {result['error']}")
    else:
        st.success("Compilation successful!")
        st.write(f"### System Name: {result['system_name']}")
        st.write("### Coordinates:")
        st.write(result['coordinates'])
        st.write("### Equations of Motion:")
        st.write(result['equations'])

        # Show AST nodes for debugging
        st.write("### AST Nodes (Debug):")
        for node in compiler.ast:
            st.json({"type": type(node).__name__, **vars(node)})

        # Run simulation
        st.write("### Running Simulation...")
        solution = compiler.simulate()
        if solution['success']:
            st.write("Simulation successful! Trajectory:")
            st.line_chart(solution['y'].T)
            st.write("Time:", solution['t'])

            # Optionally visualize energy and phase space
            st.write("### Energy and Phase Space Plots:")
            compiler.plot_energy(solution)
            compiler.plot_phase_space(solution)
        else:
            st.error(f"Simulation failed: {solution.get('error', 'Unknown error')}")
