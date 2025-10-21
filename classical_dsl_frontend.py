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
import streamlit.components.v1 as components
from math import isfinite

# Import your backend compiler/simulator
from complete_physics_dsl import *

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
        variables = getattr(compiler, 'variables', {})

        st.markdown(f"<div class='system-card'><h4>{system_name}</h4><div class='small-muted'>Coordinates: {', '.join(coordinates) if coordinates else '—'}</div></div>", unsafe_allow_html=True)

        st.subheader("Equations of Motion")
        if equations:
            for q, eq in equations.items():
                st.markdown(f"<div class='equation-display'><strong>{q}</strong><br>{str(eq)}</div>", unsafe_allow_html=True)
        else:
            st.warning("No equations derived — check your Lagrangian and coordinate definitions.")

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

            # Parameter controls
            with st.expander('Parameters (override)', expanded=False):
                param_changed = False
                if simulator and hasattr(simulator, 'parameters'):
                    params = dict(simulator.parameters)
                    new_params = {}
                    for k, v in params.items():
                        try:
                            val = float(v)
                        except Exception:
                            # Skip non-numeric parameters
                            new_params[k] = v
                            continue
                        new_val = st.number_input(f'{k}', value=val, step=0.1, format='%f', key=f'param_{k}')
                        new_params[k] = new_val
                        if new_val != val:
                            param_changed = True

                    if st.button('Update parameters & re-simulate'):
                        simulator.set_parameters(new_params)
                        try:
                            st.session_state.sim_result = simulator.simulate((0, t_max), num_points)
                            st.success('Re-simulation finished')
                        except Exception as e:
                            st.error(f'Re-simulation failed: {e}')

            # Advanced Animation & Analysis
            with st.expander("Animation & Analysis (advanced)", expanded=True):
                col1, col2 = st.columns([1,1])

                # Helper: create Plotly animation for pendulum
                def create_plotly_pendulum(sim, params, name='pendulum', trail=100):
                    t = sim['t']
                    y = sim['y']
                    theta = y[0]
                    l = params.get('l', 1.0)
                    x = l * np.sin(theta)
                    y_pos = -l * np.cos(theta)

                    frames = []
                    for i in range(len(t)):
                        frames.append(go.Frame(data=[
                            go.Scatter(x=[0, x[i]], y=[0, y_pos[i]], mode='lines+markers', line=dict(width=3, color='red'), marker=dict(size=8)),
                            go.Scatter(x=x[max(0, i-trail):i+1], y=y_pos[max(0, i-trail):i+1], mode='lines', line=dict(width=1, color='blue'), opacity=0.6)
                        ], name=str(i)))

                    fig = go.Figure(
                        data=[
                            go.Scatter(x=[0, x[0]], y=[0, y_pos[0]], mode='lines+markers', line=dict(width=3, color='red'), marker=dict(size=8)),
                            go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='blue'))
                        ],
                        layout=go.Layout(
                            xaxis=dict(range=[-l*1.2, l*1.2], autorange=False),
                            yaxis=dict(range=[-l*1.2, l*0.2], autorange=False),
                            title=f'{name.title()} Animation',
                            updatemenus=[dict(type='buttons', showactive=False,
                                              y=1.05, x=1.15, xanchor='right', yanchor='top',
                                              buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]),
                                                       dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])]
                        ),
                        frames=frames
                    )

                    return fig

                def create_plotly_double_pendulum(sim, params, trail=200):
                    t = sim['t']
                    y = sim['y']
                    theta1 = y[0]
                    theta1_dot = y[1]
                    theta2 = y[2] if y.shape[0] > 2 else np.zeros_like(theta1)
                    l1 = params.get('l1', 1.0)
                    l2 = params.get('l2', 1.0)

                    x1 = l1 * np.sin(theta1)
                    y1 = -l1 * np.cos(theta1)
                    x2 = x1 + l2 * np.sin(theta2)
                    y2 = y1 - l2 * np.cos(theta2)

                    max_reach = l1 + l2
                    frames = []
                    for i in range(len(t)):
                        frames.append(go.Frame(data=[
                            go.Scatter(x=[0, x1[i], x2[i]], y=[0, y1[i], y2[i]], mode='lines+markers', line=dict(width=3), marker=dict(size=6)),
                            go.Scatter(x=x1[max(0, i-trail):i+1], y=y1[max(0, i-trail):i+1], mode='lines', line=dict(width=1, color='red'), opacity=0.6),
                            go.Scatter(x=x2[max(0, i-trail):i+1], y=y2[max(0, i-trail):i+1], mode='lines', line=dict(width=1, color='blue'), opacity=0.6)
                        ], name=str(i)))

                    fig = go.Figure(
                        data=[
                            go.Scatter(x=[0, x1[0], x2[0]], y=[0, y1[0], y2[0]], mode='lines+markers', line=dict(width=3), marker=dict(size=6)),
                            go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='red')),
                            go.Scatter(x=[], y=[], mode='lines', line=dict(width=1, color='blue'))
                        ],
                        layout=go.Layout(
                            xaxis=dict(range=[-max_reach*1.1, max_reach*1.1], autorange=False),
                            yaxis=dict(range=[-max_reach*1.1, max_reach*0.2], autorange=False),
                            title='Double Pendulum Animation',
                            updatemenus=[dict(type='buttons', showactive=False,
                                              y=1.05, x=1.15, xanchor='right', yanchor='top',
                                              buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=30, redraw=True), fromcurrent=True)]),
                                                       dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])]
                        ),
                        frames=frames
                    )

                    return fig

                with col1:
                    plot_mode = st.selectbox('Animation mode', ['Matplotlib (backend)', 'Plotly (fast web)'], key='plot_mode')
                    if st.button('Show Animation (advanced)'):
                        try:
                            if plot_mode.startswith('Plotly'):
                                # Build Plotly animation based on system name
                                if system_name in ['simple_pendulum', 'pendulum'] or 'pendulum' in system_name:
                                    pfig = create_plotly_pendulum(sim_result, simulator.parameters, name=system_name)
                                    st.plotly_chart(pfig, use_container_width=True)
                                elif 'double' in system_name or system_name == 'double_pendulum':
                                    pfig = create_plotly_double_pendulum(sim_result, simulator.parameters)
                                    st.plotly_chart(pfig, use_container_width=True)
                                else:
                                    st.warning('Plotly animation not implemented for this system; falling back to Matplotlib.')
                                    # fallback to existing approach
                                    anim = compiler.animate(sim_result)
                                    try:
                                        js_html = anim.to_jshtml()
                                        components.html(js_html, height=600)
                                    except Exception:
                                        st.write('Could not render Matplotlib animation inline')
                            else:
                                # Use backend Matplotlib animation if available
                                anim = compiler.animate(sim_result)
                                if anim is None:
                                    st.warning('No animation available for this system')
                                else:
                                    try:
                                        js_html = anim.to_jshtml()
                                        components.html(js_html, height=600)
                                    except Exception:
                                        try:
                                            html5 = anim.to_html5_video()
                                            components.html(html5, height=400)
                                        except Exception:
                                            st.write('Could not render Matplotlib animation inline')
                        except Exception as e:
                            st.error(f'Advanced animation failed: {e}')

                with col2:
                    # Energy and phase space using Plotly for interactivity
                    if st.button('Show Energy (interactive)'):
                        try:
                            # Compute energies for supported systems
                            params = simulator.parameters
                            t = sim_result['t']
                            y = sim_result['y']
                            if 'pendulum' in system_name:
                                theta = y[0]
                                theta_dot = y[1]
                                m = params.get('m', 1.0)
                                l = params.get('l', 1.0)
                                g = params.get('g', 9.81)
                                KE = 0.5 * m * l**2 * theta_dot**2
                                PE = m * g * l * (1 - np.cos(theta))
                                E = KE + PE
                                efig = make_subplots(rows=3, cols=1, subplot_titles=('Kinetic','Potential','Total'))
                                efig.add_trace(go.Scatter(x=t, y=KE, name='KE', line=dict(color='red')), row=1, col=1)
                                efig.add_trace(go.Scatter(x=t, y=PE, name='PE', line=dict(color='blue')), row=2, col=1)
                                efig.add_trace(go.Scatter(x=t, y=E, name='Total', line=dict(color='green')), row=3, col=1)
                                efig.update_layout(height=800, template='plotly_dark')
                                st.plotly_chart(efig, use_container_width=True)
                            elif 'double' in system_name:
                                theta1, theta1_dot, theta2, theta2_dot = y[0], y[1], y[2], y[3]
                                m1 = params.get('m1', 1.0); m2 = params.get('m2', 1.0)
                                l1 = params.get('l1', 1.0); l2 = params.get('l2', 1.0)
                                g = params.get('g', 9.81)
                                KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                                KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
                                KE = KE1 + KE2
                                PE1 = -m1 * g * l1 * np.cos(theta1)
                                PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                                PE = PE1 + PE2
                                E = KE + PE
                                efig = go.Figure()
                                efig.add_trace(go.Scatter(x=t, y=E, name='Total Energy', line=dict(color='green')))
                                efig.update_layout(height=600, template='plotly_dark')
                                st.plotly_chart(efig, use_container_width=True)
                            else:
                                st.warning('Energy plotting not implemented for this system')
                        except Exception as e:
                            st.error(f'Interactive energy plotting failed: {e}')

                    if st.button('Show Phase Space (interactive)'):
                        try:
                            t = sim_result['t']
                            y = sim_result['y']
                            if y.shape[0] >= 2:
                                pos = y[0]
                                vel = y[1]
                                pfig = go.Figure()
                                pfig.add_trace(go.Scatter(x=pos, y=vel, mode='lines', line=dict(color='cyan')))
                                pfig.add_trace(go.Scatter(x=[pos[0]], y=[vel[0]], mode='markers', marker=dict(color='green', size=10), name='Start'))
                                pfig.add_trace(go.Scatter(x=[pos[-1]], y=[vel[-1]], mode='markers', marker=dict(color='red', size=10), name='End'))
                                pfig.update_layout(height=600, template='plotly_dark', title='Phase Space')
                                st.plotly_chart(pfig, use_container_width=True)
                            else:
                                st.warning('Not enough coordinates for phase space')
                        except Exception as e:
                            st.error(f'Interactive phase space failed: {e}')

            # Animation and additional analysis
            with st.expander("Animation & Analysis", expanded=True):
                col1, col2, col3 = st.columns([1,1,1])

                with col1:
                    if st.button('Show Animation'):
                        try:
                            # Create animation using the compiler's visualizer
                            if hasattr(compiler, 'animate'):
                                anim = compiler.animate(sim_result)
                            else:
                                anim = None

                            if anim is None:
                                st.warning('No animation available for this system')
                            else:
                                # Try JS/html embedding first
                                try:
                                    js_html = anim.to_jshtml()
                                    components.html(js_html, height=600)
                                except Exception as e_js:
                                    # Fallback: try HTML5 video
                                    try:
                                        html5 = anim.to_html5_video()
                                        components.html(html5, height=400)
                                    except Exception as e_vid:
                                        # Final fallback: render last frame as static image
                                        try:
                                            import io
                                            buf = io.BytesIO()
                                            anim._fig.savefig(buf, format='png')
                                            buf.seek(0)
                                            st.image(buf)
                                        except Exception as e_img:
                                            st.error(f'Could not render animation (js: {e_js}, mp4: {e_vid}, img: {e_img})')
                        except Exception as e:
                            st.error(f'Animation generation failed: {e}')

                with col2:
                    if st.button('Show Energy Plots'):
                        try:
                            # Reuse visualizer energy plotting and display the matplotlib figure(s)
                            compiler.visualizer.plot_energy(sim_result, compiler.simulator.parameters, compiler.system_name)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.error(f'Energy plotting failed: {e}')

                with col3:
                    if st.button('Show Phase Space'):
                        try:
                            compiler.visualizer.plot_phase_space(sim_result, 0)
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.error(f'Phase-space plotting failed: {e}')

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

            # Animation export
            st.markdown('---')
            st.subheader('Animation Export')

            # ffmpeg availability
            import shutil, tempfile, os
            ffmpeg_available = shutil.which('ffmpeg') is not None
            if ffmpeg_available:
                st.success('ffmpeg found — MP4 export available')
            else:
                st.warning('ffmpeg not found — MP4 export may fail. GIF fallback is available.')

            # Helper to handle export and cleanup
            def do_export(ext: str, fps: int):
                tmpdir = tempfile.gettempdir()
                filename = f"{system_name}_animation{ext}"
                out_path = os.path.join(tmpdir, filename)
                try:
                    with st.spinner(f'Generating {filename} — this may take a few seconds...'):
                        compiler.export_animation(sim_result, out_path, fps=fps)

                    with open(out_path, 'rb') as f:
                        data = f.read()

                    # Display inline preview
                    if ext == '.mp4':
                        st.video(data)
                        mime = 'video/mp4'
                    else:
                        st.image(data)
                        mime = 'image/gif'

                    st.download_button(f'Download {filename}', data, file_name=filename, mime=mime)

                except Exception as e:
                    st.error(f'Export failed: {e}')
                finally:
                    # Attempt cleanup
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass

            if st.button('Export animation as MP4'):
                if not ffmpeg_available:
                    st.warning('ffmpeg is not available — the app will try GIF fallback automatically if MP4 fails.')
                do_export('.mp4', fps=30)

            if st.button('Export animation as GIF'):
                do_export('.gif', fps=20)
