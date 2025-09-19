import streamlit as st
from complete_physics_dsl import *

st.set_page_config(page_title="Physics DSL Compiler", layout="wide")
st.title("Physics DSL Compiler")

EXAMPLE_DSL = r"""
\system{simple_pendulum}
\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}
\define{\op{kinetic}(m, l, theta_dot) = 0.5 * m * l^2 * theta_dot^2}
\define{\op{potential}(m, g, l, theta) = m * g * l * (1 - \cos{theta})}
\lagrangian{kinetic(m, l, \dot{theta}) - potential(m, g, l, theta)}
\initial{theta=0.5, theta_dot=0}
\solve{euler_lagrange}
\animate{pendulum}
"""

dsl_input = st.text_area(
    "Enter your DSL code here:",
    value=EXAMPLE_DSL,
    height=300
)

if st.button("Compile & Simulate"):
    st.write("🚀 Compiling Physics DSL...")
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(dsl_input)
    
    if not result['success']:
        st.error(f"Compilation failed: {result['error']}")
        st.stop()
    
    st.success("Compilation successful!")
    st.write(f"### System Name: {result['system_name']}")
    st.write("### Coordinates:")
    st.write(result['coordinates'])
    st.write("### Equations of Motion (symbolic):")
    st.write(result['equations'])
    
    st.write("### AST Nodes (Debugging):")
    for node in compiler.ast:
        try:
            st.json({"type": type(node).__name__, **vars(node)})
        except Exception:
            st.write(repr(node))
    
    st.write("### Running Simulation...")
    solution = compiler.simulate()
    if solution['success']:
        st.success("Simulation successful!")
        st.write("### Simulation Trajectory:")
        st.line_chart(solution['y'].T)
        st.write("Time:", solution['t'])
    else:
        st.error(f"Simulation failed: {solution.get('error', 'Unknown error')}")
