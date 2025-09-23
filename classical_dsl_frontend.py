import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import time
from typing import Dict, Any

# Import your backend compiler/simulator
from complete_physics_dsl import *

# -----------------------------
# App config & CSS (dark mode)
# -----------------------------
st.set_page_config(page_title="Physics DSL Studio", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* App background */
    .stApp, .main, header, .block-container { background-color: #0b0f1a; color: #e6eef6 }
    /* Header */
    .main-header { background: linear-gradient(90deg,#0f1724,#12203a); padding: 18px; border-radius:12px; }
    .system-card { background:#0f1724; padding:12px; border-radius:8px; border-left:6px solid #5dd1ff; margin-bottom:12px }
    .equation-display { background:#05060a; color:#7ef1c9; padding:12px; border-radius:8px; font-family: 'Fira Code', monospace; overflow-x:auto }
    .small-muted { color:#9fb0c9; font-size:12px }
    .accent-btn { background:linear-gradient(90deg,#5dd1ff,#7b6cff); color:#071124; }
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
        "desc": "Classic pendulum — good demonstration for single-DOF",
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
    "Double Pendulum": {
        "desc": "Two-link chaotic pendulum — good for multi-DOF visuals",
        "dsl": r"""
\system{double_pendulum}
\defvar{theta1}{Angle}{rad}
\defvar{theta2}{Angle}{rad}
\defvar{m1}{Mass}{kg}
\defvar{m2}{Mass}{kg}
\defvar{l1}{Length}{m}
\defvar{l2}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}
\define{\op{T1}(m1, l1, theta1_dot) = 0.5 * m1 * l1^2 * theta1_dot^2}
\define{\op{T2}(m2, l1, l2, theta1, theta2, theta1_dot, theta2_dot) = 0.5 * m2 * (l1^2 * theta1_dot^2 + l2^2 * theta2_dot^2 + 2 * l1 * l2 * theta1_dot * theta2_dot * \cos{theta1 - theta2})}
\define{\op{V1}(m1, g, l1, theta1) = -m1 * g * l1 * \cos{theta1}}
\define{\op{V2}(m2, g, l1, l2, theta1, theta2) = -m2 * g * (l1 * \cos{theta1} + l2 * \cos{theta2})}
\lagrangian{T1(m1, l1, \dot{theta1}) + T2(m2, l1, l2, theta1, theta2, \dot{theta1}, \dot{theta2}) - V1(m1, g, l1, theta1) - V2(m2, g, l1, l2, theta1, theta2)}
\initial{theta1=1.57, theta1_dot=0, theta2=1.58, theta2_dot=0}
\solve{euler_lagrange}
\animate{double_pendulum}
"""
    },
    "Harmonic Oscillator": {
        "desc": "Spring-mass linear oscillator",
        "dsl": r"""
\system{harmonic_oscillator}
\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{SpringConstant}{N/m}
\define{\op{kinetic}(m, x_dot) = 0.5 * m * x_dot^2}
\define{\op{potential}(k, x) = 0.5 * k * x^2}
\lagrangian{kinetic(m, \dot{x}) - potential(k, x)}
\initial{x=1.0, x_dot=0}
\solve{euler_lagrange}
\animate{oscillator}
"""
    }
}

choice = st.sidebar.selectbox("Load example:", list(EXAMPLES.keys()))
if st.sidebar.button("Load into editor"):
    st.session_state.dsl = EXAMPLES[choice]["dsl"]

st.sidebar.markdown("---")

# Simulation parameters
t_max = st.sidebar.slider("Simulation time (s)", min_value=1, max_value=60, value=10)
num_points = st.sidebar.slider("Number of points", min_value=100, max_value=5000, value=1500, step=100)
use_dark_theme_plots = st.sidebar.checkbox("Plotly dark theme", True)
show_debug = st.sidebar.checkbox("Show debug panel", False)

# Quick actions
if st.sidebar.button("Clear editor"):
    st.session_state.dsl = ""

# -----------------------------
# Main layout
# -----------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("Physics DSL Studio")
    st.markdown("<div class='small-muted'>Write DSL, compile symbolic equations, run numerical sims, and visualize — all in one place.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with header_col2:
    st.markdown("\n")
    if st.button("Example: Run Loaded"):
        st.experimental_rerun()

# Two columns: editor & results
editor_col, result_col = st.columns([1, 1.4])

with editor_col:
    st.subheader("DSL Editor")
    dsl_text = st.text_area("Enter your physics DSL:", value=st.session_state.get('dsl', EXAMPLES[choice]['dsl']), height=420, key='dsl_editor')

    st.markdown("---")
    st.markdown("**Token analysis / quick debug**")
    if st.button("Analyze tokens"):
        try:
            tokens = tokenize(dsl_text)
            token_df = pd.DataFrame([{'type': t.type, 'value': t.value, 'line': t.line, 'col': t.column} for t in tokens])
            st.dataframe(token_df)
            st.success(f"Found {len(tokens)} tokens")
        except Exception as e:
            st.error(f"Tokenization failed: {e}")

    st.markdown("---")
    st.markdown("**Compile & simulate**")
    run_now = st.button("Compile & Simulate")

with result_col:
    st.subheader("Results & Visualization")

    # Placeholder boxes
    system_info_box = st.container()
    equations_box = st.container()
    viz_box = st.container()
    controls_box = st.container()

# -----------------------------
# Main actions: Compile & Simulate
# -----------------------------
if run_now:
    compiler = PhysicsCompiler()
    status = system_info_box
    with status:
        st.info("Compiling DSL and deriving equations — this may take a few seconds...")

    try:
        compile_result = compiler.compile_dsl(dsl_text)
    except Exception as e:
        st.error(f"Compilation pipeline raised an exception: {e}")
        compile_result = {'success': False, 'error': str(e)}

    if not compile_result.get('success'):
        st.error(f"Compilation failed: {compile_result.get('error', 'Unknown error')}")
    else:
        # Extract results
        system_name = compile_result.get('system_name', 'unnamed')
        coordinates = compile_result.get('coordinates', [])
        equations = compile_result.get('equations', {})

        # Show system card
        with system_info_box:
            st.markdown(f"<div class='system-card'><h4> {system_name}</h4><div class='small-muted'>Coordinates: {', '.join(coordinates) if coordinates else '—'}</div></div>", unsafe_allow_html=True)

        # Show equations
        with equations_box:
            st.subheader("Equations of Motion")
            if equations:
                for q, eq in equations.items():
                    st.markdown(f"<div class='equation-display'><strong>{q}</strong><br> {str(eq)}</div>", unsafe_allow_html=True)
            else:
                st.warning("No equations derived — check your Lagrangian and coordinate definitions.")

        # Setup simulator
        simulator = compile_result.get('simulator')
        # Ensure simulator has initial conditions & parameters
        try:
            # Basic parameters (user editable)
            st.sidebar.markdown('---')
            st.sidebar.subheader('Simulation Parameters (override)')

            # Provide defaults and let user input JSON for parameters
            default_params = getattr(simulator, 'parameters', {}) or {}
            params_str = st.sidebar.text_area('Parameters (JSON)', value=json.dumps(default_params, indent=2), height=120)
            try:
                params = json.loads(params_str) if params_str.strip() else {}
            except Exception as e:
                st.sidebar.error(f'Invalid JSON: {e}')
                params = default_params

            simulator.set_parameters(params)

            # Initial conditions
            default_ic = getattr(simulator, 'initial_conditions', {}) or {}
            ic_str = st.sidebar.text_area('Initial conditions (JSON)', value=json.dumps(default_ic, indent=2), height=120)
            try:
                ic = json.loads(ic_str) if ic_str.strip() else {}
            except Exception as e:
                st.sidebar.error(f'Invalid JSON: {e}')
                ic = default_ic

            simulator.set_initial_conditions(ic)

            # Compile symbolic accelerations to numerical functions
            # The compiler should have produced 'equations' as a dict of sympy expressions
            symbolic_engine = compiler.symbolic
            accel_map = symbolic_engine.solve_for_accelerations(list(equations.values()), [c.replace('_dot','') for c in coordinates]) if equations else {}
            # If the backend already compiled inside compiler, try to use that
            if accel_map:
                simulator.compile_equations(accel_map, [c.replace('_dot','') for c in coordinates])
            else:
                # Fallback: try to ask compiler for its simulator (already compiled)
                try:
                    simulator = compile_result.get('simulator', simulator)
                except:
                    pass

        except Exception as e:
            st.error(f"Simulator setup failed: {e}")

        # Run numerical simulation
        with viz_box:
            st.subheader("Numerical Simulation")
            placeholder = st.empty()
            with st.spinner('Running numerical integration...'):
                try:
                    sim_result = simulator.simulate((0, t_max), num_points)
                except Exception as e:
                    st.error(f"Simulation crashed: {e}")
                    sim_result = {'success': False, 'error': str(e)}

            if not sim_result.get('success'):
                st.error(f"Simulation failed: {sim_result.get('error', 'Unknown')}")
            else:
                t = sim_result['t']
                y = sim_result['y']
                coords = sim_result.get('coordinates', simulator.coordinates if hasattr(simulator, 'coordinates') else [])

                st.success('Simulation finished')

                # Interactive time slider + play/pause
                controls = controls_box
                with controls:
                    st.markdown('**Playback Controls**')
                    play = st.button('▶ Play')
                    pause = st.button('⏸ Pause')
                    time_idx = st.slider('Frame', 0, len(t)-1, 0, format='%d')

                # Build figure(s)
                fig = make_subplots(rows=2, cols=2, subplot_titles=("Position","Velocity","Phase Space","Energy"), specs=[[{}, {}],[{}, {}]])

                colors = px.colors.qualitative.Vivid

                # Positions and velocities
                for i, coord in enumerate(coords):
                    pos = y[2*i]
                    vel = y[2*i + 1]
                    fig.add_trace(go.Scatter(x=t, y=pos, name=f"{coord}(t)", mode='lines', legendgroup=f'g{i}'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=t, y=vel, name=f"{coord}_dot(t)", mode='lines', line=dict(dash='dash'), legendgroup=f'g{i}'), row=1, col=2)

                # Phase space for first coordinate
                if len(coords) > 0:
                    fig.add_trace(go.Scatter(x=y[0], y=y[1], mode='lines', name='Phase'), row=2, col=1)

                # Energy (best-effort from backend visualizer)
                try:
                    energy_fig = compiler.visualizer.plot_energy(sim_result, simulator.parameters if hasattr(simulator, 'parameters') else {}, system_name)
                except Exception:
                    energy_fig = None

                if use_dark_theme_plots:
                    fig.update_layout(template='plotly_dark', paper_bgcolor='#0b0f1a', plot_bgcolor='#071025')
                else:
                    fig.update_layout(template='plotly', paper_bgcolor='#ffffff')

                fig.update_layout(height=700, showlegend=True, title_text=f"Analysis: {system_name}")

                st.plotly_chart(fig, use_container_width=True)

                # Frame-specific 2D visualization for selected frame
                with st.expander('Frame preview & 3D (if available)'):
                    frame_idx = time_idx
                    col_a, col_b = st.columns([1,1])
                    with col_a:
                        st.markdown(f"**Frame {frame_idx} (t={t[frame_idx]:.3f}s)**")
                        table = {f'{coord}': list(y[2*i][frame_idx:frame_idx+1])[0] for i, coord in enumerate(coords)}
                        st.json(table)
                    with col_b:
                        # Simple 2D pendulum drawing for pendulum-like systems
                        if 'pendulum' in system_name:
                            # assume first coordinate is angle
                            theta = float(y[0][frame_idx])
                            l = float(simulator.parameters.get('l', 1.0))
                            x = l * np.sin(theta)
                            y_pos = -l * np.cos(theta)

                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=[0, x], y=[0, y_pos], mode='lines+markers', line=dict(width=6)))
                            fig2.update_layout(height=350, template='plotly_dark', xaxis=dict(range=[-l*1.5, l*1.5]), yaxis=dict(range=[-l*1.5, l*0.5]))
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info('3D preview not available for this system type yet')

                # Exporters
                st.markdown('---')
                st.subheader('Export')
                # Trajectory CSV
                traj_df = pd.DataFrame({'time': t})
                for i, coord in enumerate(coords):
                    traj_df[coord] = y[2*i]
                    traj_df[f'{coord}_dot'] = y[2*i + 1]

                csv = traj_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Trajectories CSV', csv, file_name=f'{system_name}_trajectories.csv', mime='text/csv')

                # Equations LaTeX
                latex = '\\begin{align}\n'
                for q, eq in equations.items():
                    latex += f"\\ddot{{{q}}} &= {str(eq)} \\\\n"
                latex += '\\end{align}'
                st.download_button('Download Equations (LaTeX)', latex, file_name=f'{system_name}_equations.tex')

                # System JSON
                sys_json = json.dumps({'system_name': system_name, 'coordinates': coords, 'parameters': simulator.parameters, 'initial_conditions': simulator.initial_conditions, 't_max': t_max, 'num_points': num_points}, indent=2, default=str)
                st.download_button('Download System JSON', sys_json, file_name=f'{system_name}_config.json')

                # Debug panel
                if show_debug:
                    with st.expander('🔧 Debug Info'):
                        st.subheader('Compiler AST')
                        try:
                            st.json([repr(n) for n in compiler.ast])
                        except Exception as e:
                            st.write(f'AST unavailable: {e}')

                        st.subheader('Symbol map')
                        try:
                            st.json({k: str(v) for k, v in compiler.symbolic.symbol_map.items()})
                        except Exception:
                            pass

                        st.subheader('Simulator internals')
                        try:
                            st.write(f"State vars: {getattr(simulator, 'state_vars', [])}")
                            st.write(f"Equations compiled: {list(getattr(simulator, 'equations', {}).keys())}")
                        except Exception:
                            pass

# Small helpful footer
st.markdown('---')
st.markdown("<div class='small-muted'>Built for rapid prototyping — reach out if you want custom visualization widgets, camera-follow animations, or export to MP4/GIF.</div>", unsafe_allow_html=True)
