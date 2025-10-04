import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any

# Import your backend compiler/simulator
from complete_physics_dsl import *
from matlab_export_generator import *

# -----------------------------
# App config & CSS (dark mode)
# -----------------------------
st.set_page_config(page_title="Physics DSL Studio", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp, .main, header, .block-container { background-color: #0b0f1a; color: #e6eef6 }
    .main-header { background: linear-gradient(90deg,#0f1724,#12203a); padding: 18px; border-radius:12px; }
    .system-card { background:#0f1724; padding:12px; border-radius:8px; border-left:6px solid #5dd1ff; margin-bottom:12px }
    .equation-display { background:#05060a; color:#7ef1c9; padding:12px; border-radius:8px; font-family: 'Fira Code', monospace; overflow-x:auto }
    .small-muted { color:#9fb0c9; font-size:12px }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar: examples + settings
# -----------------------------
st.sidebar.title("Physics DSL Studio")

EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Simple Pendulum": {
        "dsl": r"""
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
    },
    "Harmonic Oscillator": {
        "dsl": r"""
\system{harmonic_oscillator}
\defvar{x}{Displacement}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\define{\op{kinetic}(m, x_dot) = 0.5 * m * x_dot^2}
\define{\op{potential}(k, x) = 0.5 * k * x^2}
\lagrangian{kinetic(m, \dot{x}) - potential(k, x)}
\initial{x=1, x_dot=0}
\solve{euler_lagrange}
\animate{oscillator}
"""
    },
    "Double Pendulum": {
        "dsl": r"""
\system{double_pendulum}
\defvar{theta1}{Angle1}{rad}
\defvar{theta2}{Angle2}{rad}
\defvar{m1}{Mass1}{kg}
\defvar{m2}{Mass2}{kg}
\defvar{l1}{Length1}{m}
\defvar{l2}{Length2}{m}
\defvar{g}{Gravity}{m/s^2}
\define{\op{kinetic}(m1, m2, l1, l2, theta1_dot, theta2_dot, theta1, theta2) =
 0.5*m1*l1^2*theta1_dot^2 + 0.5*m2*(l1^2*theta1_dot^2 + l2^2*theta2_dot^2 +
 2*l1*l2*theta1_dot*theta2_dot*cos(theta1-theta2))}
\define{\op{potential}(m1, m2, g, l1, l2, theta1, theta2) =
 -m1*g*l1*cos(theta1) - m2*g*(l1*cos(theta1)+l2*cos(theta2))}
\lagrangian{kinetic(m1,m2,l1,l2,\dot{theta1},\dot{theta2},theta1,theta2) - potential(m1,m2,g,l1,l2,theta1,theta2)}
\initial{theta1=0.5, theta2=1.0, theta1_dot=0, theta2_dot=0}
\solve{euler_lagrange}
\animate{double_pendulum}
"""
    },
    "Damped Oscillator": {
        "dsl": r"""
\system{damped_oscillator}
\defvar{x}{Displacement}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{Spring Constant}{N/m}
\defvar{c}{Damping}{Ns/m}
\define{\op{kinetic}(m, x_dot) = 0.5 * m * x_dot^2}
\define{\op{potential}(k, x) = 0.5 * k * x^2}
\define{\op{dissipation}(c, x_dot) = 0.5 * c * x_dot^2}
\lagrangian{kinetic(m, \dot{x}) - potential(k, x) - dissipation(c, \dot{x})}
\initial{x=1, x_dot=0}
\solve{euler_lagrange}
\animate{damped}
"""
    }
}

choice = st.sidebar.selectbox("Load example:", list(EXAMPLES.keys()))
if st.sidebar.button("Load into editor"):
    st.session_state.dsl = EXAMPLES[choice]["dsl"]

t_max = st.sidebar.slider("Simulation time (s)", 1, 60, 10)
num_points = st.sidebar.slider("Number of points", 100, 5000, 1500, step=100)

# -----------------------------
# Main layout
# -----------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Physics DSL Studio")
st.markdown("<div class='small-muted'>Write DSL, compile symbolic equations, run numerical simulations, and visualize — all in one place.</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Two columns: editor & results
editor_col, result_col = st.columns([1, 1.4])

with editor_col:
    st.subheader("DSL Editor")
    dsl_text = st.text_area("Enter your physics DSL:", value=st.session_state.get('dsl', EXAMPLES[choice]['dsl']), height=420, key='dsl_editor')

    st.markdown("---")
    run_now = st.button("Compile & Simulate")


with result_col:
    st.subheader("Results & Visualization")
    
    if run_now:
        compiler = PhysicsCompiler()
        try:
            compile_result = compiler.compile_dsl(dsl_text)
        except Exception as e:
            st.error(f"Compilation pipeline raised an exception: {e}")
            compile_result = {'success': False, 'error': str(e)}
            
        if not compile_result.get('success'):
            st.error(f"Compilation failed: {compile_result.get('error', 'Unknown error')}")
        else:
            st.session_state.compile_result = compile_result
            st.session_state.compiler = compiler

    if 'compile_result' in st.session_state and st.session_state.compile_result.get('success'):
        compile_result = st.session_state.compile_result
        compiler = st.session_state.compiler
        
        system_name = compile_result.get('system_name', 'unnamed')
        coordinates = compile_result.get('coordinates', [])
        equations = compile_result.get('equations', {})
        simulator = compile_result.get('simulator')

        st.markdown(f"<div class='system-card'><h4>{system_name}</h4><div class='small-muted'>Coordinates: {', '.join(coordinates) if coordinates else '—'}</div></div>", unsafe_allow_html=True)

        st.subheader("Equations of Motion")
        if equations:
            for q, eq in equations.items():
                st.markdown(f"<div class='equation-display'><strong>{q}</strong><br>{str(eq)}</div>", unsafe_allow_html=True)
        else:
            st.warning("No equations derived — check your Lagrangian and coordinate definitions.")

        # Run numerical simulation (only once)
        if 'sim_result' not in st.session_state or run_now:
            try:
                sim_result = simulator.simulate((0, t_max), num_points)
                st.session_state.sim_result = sim_result
            except Exception as e:
                st.error(f"Simulation crashed: {e}")
                st.session_state.sim_result = {'success': False, 'error': str(e)}

        sim_result = st.session_state.sim_result

        if sim_result.get('success'):
            t = sim_result['t']
            y = sim_result['y']
            coords = sim_result.get('coordinates', simulator.coordinates if hasattr(simulator, 'coordinates') else [])

            st.success('Simulation finished')

            # Static plots
            fig = make_subplots(rows=2, cols=2, subplot_titles=("Position","Velocity","Phase Space","Energy"))
            for i, coord in enumerate(coords):
                fig.add_trace(go.Scatter(x=t, y=y[2*i], name=f"{coord}(t)"), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=y[2*i+1], name=f"{coord}_dot(t)", line=dict(dash='dash')), row=1, col=2)
            if len(coords) > 0:
                fig.add_trace(go.Scatter(x=y[0], y=y[1], name="Phase"), row=2, col=1)
            fig.update_layout(template='plotly_dark', height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Export options
            st.subheader("Export")

            # CSV Export
            traj_df = pd.DataFrame({'time': t})
            for i, coord in enumerate(coords):
                traj_df[coord] = y[2*i]
                traj_df[f"{coord}_dot"] = y[2*i+1]

            csv = traj_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Trajectories CSV', csv, file_name=f'{system_name}_trajectories.csv', mime='text/csv')

            # LaTeX Export
            latex = '\\\\begin{align}\\n'
            for q, eq in equations.items():
                latex += f"\\\\ddot{{{q}}} &= {str(eq)} \\\\\\n"
            latex += '\\\\end{align}'
            st.download_button('Download Equations (LaTeX)', latex, file_name=f'{system_name}_equations.tex')

            # JSON Export
            sys_json = json.dumps({
                'system_name': system_name,
                'coordinates': coords,
                'parameters': simulator.parameters,
                'initial_conditions': simulator.initial_conditions,
                't_max': t_max,
                'num_points': num_points
            }, indent=2, default=str)
            st.download_button('Download System JSON', sys_json, file_name=f'{system_name}_config.json')
           
            # MATLAB Export
            st.markdown("---")
            if st.button('🔬 Generate MATLAB Validation Script'):
                try:
                    matlab_file = compiler.export_to_matlab(equations)
                    with open(matlab_file, 'r') as f:
                        matlab_code = f.read()
                    st.session_state.matlab_code = matlab_code
                    st.session_state.matlab_filename = matlab_file
                except Exception as e:
                    st.error(f'MATLAB export failed: {e}')
           
            # Show download button if MATLAB code was generated
            if 'matlab_code' in st.session_state:
                st.download_button(
                    '📥 Download MATLAB Script',
                    st.session_state.matlab_code,
                    file_name=st.session_state.matlab_filename,
                    mime='text/plain'
                )
                st.success(f'✓ Generated: {st.session_state.matlab_filename}')
