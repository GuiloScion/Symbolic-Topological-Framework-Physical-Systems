import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from complete_physics_dsl import *

st.set_page_config(page_title="Advanced Physics DSL Compiler", layout="wide", page_icon="⚛️")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.system-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 5px solid #667eea;
    margin: 1rem 0;
}
.equation-display {
    background: #1e1e1e;
    color: #00ff00;
    padding: 1rem;
    border-radius: 8px;
    font-family: monospace;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚛️ Advanced Physics DSL Compiler</h1>
    <p>Symbolic-Topological Framework for Physical Systems</p>
    <p><em>Automatic derivation of equations of motion from Lagrangian mechanics</em></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for examples and settings
st.sidebar.title("🎛️ System Examples")

example_systems = {
    "Simple Pendulum": {
        "description": "Classic pendulum with gravitational restoring force",
        "complexity": "Basic",
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
        "description": "Two coupled pendulums exhibiting chaotic behavior",
        "complexity": "Advanced",
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
        "description": "Spring-mass system with Hooke's law",
        "complexity": "Basic",
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

selected_example = st.sidebar.selectbox("Choose Example System:", list(example_systems.keys()))

if st.sidebar.button("Load Example"):
    st.session_state.dsl_input = example_systems[selected_example]["dsl"]

# Display example info
if selected_example:
    example = example_systems[selected_example]
    st.sidebar.markdown(f"""
    **Description:** {example['description']}
    
    **Complexity:** {example['complexity']}
    """)

# Simulation parameters
st.sidebar.title("🔧 Simulation Settings")
t_max = st.sidebar.slider("Simulation Time (s)", 1, 20, 10)
num_points = st.sidebar.slider("Number of Points", 100, 2000, 1000)
show_debug = st.sidebar.checkbox("Show Debug Information", False)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 DSL Input")
    
    # Get DSL input
    dsl_input = st.text_area(
        "Enter your Physics DSL code:",
        value=st.session_state.get('dsl_input', example_systems[selected_example]["dsl"]),
        height=400,
        help="Write your physics system using the DSL syntax. See examples in the sidebar."
    )
    
    # Token analysis button
    if st.button("🔍 Analyze Tokens"):
        tokens = tokenize(dsl_input)
        st.subheader("Token Analysis")
        
        token_df = pd.DataFrame([
            {"Token": token.type, "Value": token.value, "Line": token.line, "Column": token.column}
            for token in tokens
        ])
        st.dataframe(token_df)
        
        # Token type distribution
        token_counts = token_df['Token'].value_counts()
        fig = px.pie(values=token_counts.values, names=token_counts.index, 
                    title="Token Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🚀 Compilation Results")
    
    if st.button("⚡ Compile & Simulate", type="primary"):
        with st.spinner("🔄 Compiling Physics DSL..."):
            compiler = PhysicsCompiler()
            result = compiler.compile_dsl(dsl_input)
        
        if not result['success']:
            st.error(f"❌ Compilation failed: {result['error']}")
            st.stop()
        
        st.success("✅ Compilation successful!")
        
        # System information
        st.markdown(f"""
        <div class="system-card">
            <h4>📊 System: {result['system_name']}</h4>
            <p><strong>Coordinates:</strong> {', '.join(result['coordinates'])}</p>
            <p><strong>Degrees of Freedom:</strong> {len(result['coordinates'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show equations of motion
        st.subheader("⚡ Derived Equations of Motion")
        st.markdown("*Automatically derived using Euler-Lagrange equations:*")
        
        for coord, equation in result['equations'].items():
            st.markdown(f"""
            <div class="equation-display">
                <strong>{coord}:</strong><br>
                {str(equation)}
            </div>
            """, unsafe_allow_html=True)
        
        # Debug information
        if show_debug:
            st.subheader("🔧 Debug Information")
            with st.expander("AST Structure"):
                for i, node in enumerate(compiler.ast):
                    st.write(f"**Node {i}:** {type(node).__name__}")
                    try:
                        st.json(vars(node))
                    except:
                        st.write(repr(node))
            
            with st.expander("Variable Definitions"):
                st.json(compiler.variables)
            
            with st.expander("Function Definitions"):
                st.json({k: str(v) for k, v in compiler.definitions.items()})
        
        # Run simulation
        st.subheader("📈 Numerical Simulation")
        
        with st.spinner("🔄 Running simulation..."):
            solution = compiler.simulate((0, t_max), num_points)
        
        if not solution['success']:
            st.error(f"❌ Simulation failed: {solution.get('error', 'Unknown error')}")
            st.stop()
        
        st.success("✅ Simulation completed!")
        
        # Create comprehensive visualizations
        st.subheader("📊 Results Visualization")
        
        # Time series plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Position vs Time", "Velocity vs Time", "Phase Space", "Energy Analysis"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot trajectories
        coords = result['coordinates']
        t = solution['t']
        y = solution['y']
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, coord in enumerate(coords):
            # Position
            fig.add_trace(
                go.Scatter(x=t, y=y[2*i], name=f"{coord}(t)", 
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=1
            )
            
            # Velocity
            fig.add_trace(
                go.Scatter(x=t, y=y[2*i+1], name=f"{coord}_dot(t)", 
                          line=dict(color=colors[i % len(colors)], dash='dash')),
                row=1, col=2
            )
        
        # Phase space (first coordinate)
        if len(coords) > 0:
            fig.add_trace(
                go.Scatter(x=y[0], y=y[1], mode='lines+markers',
                          name="Phase Space", line=dict(color='purple')),
                row=2, col=1
            )
        
        # Energy calculation (simplified for demo)
        if result['system_name'] == 'simple_pendulum':
            # Calculate energies for pendulum
            theta = y[0]
            theta_dot = y[1]
            
            # Assume default parameters
            m, l, g = 1.0, 1.0, 9.81
            
            KE = 0.5 * m * l**2 * theta_dot**2
            PE = m * g * l * (1 - np.cos(theta))
            E_total = KE + PE
            
            fig.add_trace(
                go.Scatter(x=t, y=KE, name="Kinetic Energy", 
                          line=dict(color='red')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=t, y=PE, name="Potential Energy", 
                          line=dict(color='blue')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=t, y=E_total, name="Total Energy", 
                          line=dict(color='green', width=3)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Complete System Analysis")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D animation for pendulum systems
        if 'pendulum' in result['system_name']:
            st.subheader("🎬 3D Animation")
            
            # Create animation frames
            frames = []
            for i in range(0, len(t), max(1, len(t)//50)):  # 50 frames max
                frame_data = []
                
                if result['system_name'] == 'simple_pendulum':
                    theta = y[0][i]
                    l = 1.0  # Default length
                    x = l * np.sin(theta)
                    y_pos = -l * np.cos(theta)
                    
                    # Pendulum rod and bob
                    frame_data.append(
                        go.Scatter3d(
                            x=[0, x], y=[0, y_pos], z=[0, 0],
                            mode='lines+markers',
                            line=dict(color='black', width=8),
                            marker=dict(size=[0, 15], color=['black', 'red'])
                        )
                    )
                
                elif result['system_name'] == 'double_pendulum':
                    theta1, theta2 = y[0][i], y[2][i]
                    l1, l2 = 1.0, 1.0  # Default lengths
                    
                    x1 = l1 * np.sin(theta1)
                    y1 = -l1 * np.cos(theta1)
                    x2 = x1 + l2 * np.sin(theta2)
                    y2 = y1 - l2 * np.cos(theta2)
                    
                    # Double pendulum
                    frame_data.append(
                        go.Scatter3d(
                            x=[0, x1, x2], y=[0, y1, y2], z=[0, 0, 0],
                            mode='lines+markers',
                            line=dict(color='black', width=8),
                            marker=dict(size=[0, 12, 12], color=['black', 'red', 'blue'])
                        )
                    )
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            # Create figure with frames
            anim_fig = go.Figure(
                data=frames[0].data if frames else [],
                frames=frames
            )
            
            # Add play/pause buttons
            anim_fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True}],
                         "label": "Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate", "transition": {"duration": 0}}],
                         "label": "Pause", "method": "animate"}
                    ],
                    "direction": "left", "pad": {"r": 10, "t": 87},
                    "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
                }],
                scene=dict(
                    xaxis=dict(range=[-3, 3]),
                    yaxis=dict(range=[-3, 1]),
                    zaxis=dict(range=[-0.1, 0.1]),
                    aspectmode='cube'
                ),
                title="3D Pendulum Animation"
            )
            
            st.plotly_chart(anim_fig, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Results")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            # Export trajectory data
            trajectory_df = pd.DataFrame({
                'time': t,
                **{f"{coord}": y[2*i] for i, coord in enumerate(coords)},
                **{f"{coord}_dot": y[2*i+1] for i, coord in enumerate(coords)}
            })
            
            csv = trajectory_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Trajectory CSV",
                data=csv,
                file_name=f"{result['system_name']}_trajectory.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # Export equations as LaTeX
            equations_latex = "\\begin{align}\n"
            for coord, eq in result['equations'].items():
                equations_latex += f"\\ddot{{{coord}}} &= {str(eq)} \\\\\n"
            equations_latex += "\\end{align}"
            
            st.download_button(
                label="📝 Download Equations (LaTeX)",
                data=equations_latex,
                file_name=f"{result['system_name']}_equations.tex",
                mime="text/plain"
            )
        
        with col_export3:
            # Export system parameters
            system_json = {
                "system_name": result['system_name'],
                "coordinates": result['coordinates'],
                "variables": compiler.variables,
                "initial_conditions": compiler.initial_conditions,
                "simulation_time": t_max,
                "num_points": num_points
            }
            
            import json
            system_json_str = json.dumps(system_json, indent=2, default=str)
            st.download_button(
                label="⚙️ Download System Config",
                data=system_json_str,
                file_name=f"{result['system_name']}_config.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>Physics DSL Compiler</h4>
    <p>Symbolic-Topological Framework for Physical Systems</p>
    <p><em>Built with Streamlit • Powered by SymPy • Created by Noah Parsons</em></p>
</div>
""", unsafe_allow_html=True)