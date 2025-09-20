import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from complete_physics_dsl import *
from scipy.stats import linregress
import seaborn as sns

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
.chaos-metric {
    background: linear-gradient(45deg, #ff6b6b, #ff8e53);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin: 0.5rem 0;
}
.parameter-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'compiled_system' not in st.session_state:
    st.session_state.compiled_system = None
if 'last_solution' not in st.session_state:
    st.session_state.last_solution = None

# Header
st.markdown("""
<div class="main-header">
    <h1>⚛️ Advanced Physics DSL Compiler</h1>
    <p>Symbolic-Topological Framework for Physical Systems</p>
    <p><em>Complete chaos analysis, parameter studies, and custom system design</em></p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar with tabs
sidebar_tabs = st.sidebar.radio("Navigation", ["🎛️ Examples", "🔧 Parameters", "📊 Analysis", "🏗️ Builder"])

# EXAMPLES TAB
if sidebar_tabs == "🎛️ Examples":
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
\lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
\initial{theta=0.5, theta_dot=0}
\solve{euler_lagrange}
\animate{pendulum}
""",
            "parameters": {"m": 1.0, "l": 1.0, "g": 9.81}
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
\lagrangian{0.5 * (m1 + m2) * l1^2 * \dot{theta1}^2 + 0.5 * m2 * l2^2 * \dot{theta2}^2 + m2 * l1 * l2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2} - (m1 + m2) * g * l1 * \cos{theta1} - m2 * g * l2 * \cos{theta2}}
\initial{theta1=1.57, theta1_dot=0, theta2=1.58, theta2_dot=0}
\solve{euler_lagrange}
\animate{double_pendulum}
""",
            "parameters": {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        },
        
        "Harmonic Oscillator": {
            "description": "Spring-mass system with Hooke's law",
            "complexity": "Basic",
            "dsl": r"""
\system{harmonic_oscillator}
\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{SpringConstant}{N/m}
\lagrangian{0.5 * m * \dot{x}^2 - 0.5 * k * x^2}
\initial{x=1.0, x_dot=0}
\solve{euler_lagrange}
\animate{oscillator}
""",
            "parameters": {"m": 1.0, "k": 1.0}
        },
        
        "Damped Oscillator": {
            "description": "Oscillator with damping force",
            "complexity": "Intermediate", 
            "dsl": r"""
\system{damped_oscillator}
\defvar{x}{Position}{m}
\defvar{m}{Mass}{kg}
\defvar{k}{SpringConstant}{N/m}
\defvar{c}{Damping}{Ns/m}
\lagrangian{0.5 * m * \dot{x}^2 - 0.5 * k * x^2}
\initial{x=2.0, x_dot=0}
\solve{euler_lagrange}
\animate{oscillator}
""",
            "parameters": {"m": 1.0, "k": 4.0, "c": 0.5}
        }
    }
    
    selected_example = st.sidebar.selectbox("Choose Example System:", list(example_systems.keys()))
    
    if st.sidebar.button("Load Example"):
        st.session_state.dsl_input = example_systems[selected_example]["dsl"]
        st.session_state.system_parameters = example_systems[selected_example]["parameters"]
    
    # Display example info
    if selected_example:
        example = example_systems[selected_example]
        st.sidebar.markdown(f"""
        **Description:** {example['description']}
        **Complexity:** {example['complexity']}
        """)

# PARAMETERS TAB
elif sidebar_tabs == "🔧 Parameters":
    st.sidebar.title("🔧 System Parameters")
    
    # Simulation settings
    st.sidebar.subheader("Simulation Settings")
    t_max = st.sidebar.slider("Simulation Time (s)", 1, 50, 10)
    num_points = st.sidebar.slider("Number of Points", 100, 5000, 1000)
    
    # Physical parameters (if system is loaded)
    if st.session_state.compiled_system:
        st.sidebar.subheader("Physical Parameters")
        params = st.session_state.get('system_parameters', {})
        
        for param, default_val in params.items():
            params[param] = st.sidebar.number_input(
                f"{param}", 
                value=float(default_val),
                step=0.1,
                format="%.3f"
            )
        
        st.session_state.system_parameters = params
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    show_debug = st.sidebar.checkbox("Show Debug Information", False)
    high_precision = st.sidebar.checkbox("High Precision Mode", False)
    
    # Chaos analysis settings
    st.sidebar.subheader("Chaos Analysis")
    chaos_enabled = st.sidebar.checkbox("Enable Chaos Analysis", True)
    lyapunov_time = st.sidebar.slider("Lyapunov Analysis Time", 10, 100, 50)
    sensitivity_range = st.sidebar.slider("Sensitivity Range (%)", 1, 20, 5)

# ANALYSIS TAB
elif sidebar_tabs == "📊 Analysis":
    st.sidebar.title("📊 Analysis Tools")
    
    analysis_options = st.sidebar.multiselect(
        "Select Analysis Types:",
        ["Phase Space", "Energy Conservation", "Poincaré Sections", 
         "Frequency Analysis", "Stability Analysis", "Bifurcation Diagram"],
        default=["Phase Space", "Energy Conservation"]
    )
    
    # Chaos metrics
    if st.session_state.last_solution:
        st.sidebar.subheader("Chaos Metrics")
        
        # Calculate Lyapunov exponent (simplified)
        def calculate_lyapunov_approx(solution):
            y = solution['y']
            if len(y) >= 4:  # Need at least 2 coordinates
                # Simple approximation for largest Lyapunov exponent
                dt = solution['t'][1] - solution['t'][0]
                divergence_rate = np.mean(np.abs(np.diff(y[1]))) / dt
                return np.log(divergence_rate) if divergence_rate > 0 else -np.inf
            return 0
        
        lyapunov = calculate_lyapunov_approx(st.session_state.last_solution)
        st.sidebar.metric("Lyapunov Exponent", f"{lyapunov:.4f}")
        
        if lyapunov > 0:
            st.sidebar.success("System shows chaotic behavior")
        else:
            st.sidebar.info("System appears stable")

# BUILDER TAB
elif sidebar_tabs == "🏗️ Builder":
    st.sidebar.title("🏗️ Custom System Builder")
    
    system_type = st.sidebar.selectbox(
        "System Type:",
        ["Pendulum", "Oscillator", "Central Force", "Coupled System"]
    )
    
    if system_type == "Pendulum":
        pendulum_type = st.sidebar.radio("Pendulum Type:", ["Simple", "Double", "Triple"])
        include_damping = st.sidebar.checkbox("Include Damping")
        include_driving = st.sidebar.checkbox("Include Driving Force")
        
        if st.sidebar.button("Generate Pendulum DSL"):
            # Generate DSL based on selections
            if pendulum_type == "Simple":
                dsl = r"""
\system{custom_pendulum}
\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}
"""
                if include_damping:
                    dsl += r"\defvar{b}{Damping}{Ns*m}" + "\n"
                
                dsl += r"\lagrangian{0.5 * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}"
                
            st.session_state.dsl_input = dsl

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📝 DSL Editor", "📊 Simulation", "🌪️ Chaos Analysis", "📈 Parameter Study"])

with tab1:
    st.header("📝 DSL Input & Compilation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Get DSL input
        dsl_input = st.text_area(
            "Enter your Physics DSL code:",
            value=st.session_state.get('dsl_input', example_systems["Simple Pendulum"]["dsl"]),
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
    
    with col2:
        st.subheader("🚀 Compilation Results")
        
        if st.button("⚡ Compile System", type="primary"):
            with st.spinner("🔄 Compiling Physics DSL..."):
                compiler = PhysicsCompiler()
                result = compiler.compile_dsl(dsl_input)
            
            if not result['success']:
                st.error(f"❌ Compilation failed: {result['error']}")
                st.stop()
            
            st.success("✅ Compilation successful!")
            st.session_state.compiled_system = result
            st.session_state.compiler = compiler
            
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
            for coord, equation in result['equations'].items():
                st.markdown(f"""
                <div class="equation-display">
                    <strong>{coord}:</strong><br>
                    {str(equation)}
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.header("📊 Numerical Simulation & Visualization")
    
    if not st.session_state.compiled_system:
        st.warning("Please compile a system first in the DSL Editor tab.")
    else:
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("🔄 Running simulation..."):
                # Update parameters if available
                if 'system_parameters' in st.session_state:
                    st.session_state.compiler.simulator.set_parameters(st.session_state.system_parameters)
                
                solution = st.session_state.compiler.simulate((0, t_max), num_points)
                st.session_state.last_solution = solution
            
            if not solution['success']:
                st.error(f"❌ Simulation failed: {solution.get('error', 'Unknown error')}")
                st.stop()
            
            st.success("✅ Simulation completed!")
            
            # Create comprehensive visualizations
            result = st.session_state.compiled_system
            
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
            
            # Energy calculation
            system_name = result['system_name']
            params = st.session_state.get('system_parameters', {})
            
            if 'pendulum' in system_name:
                if system_name == 'simple_pendulum':
                    theta = y[0]
                    theta_dot = y[1]
                    
                    m = params.get('m', 1.0)
                    l = params.get('l', 1.0)
                    g = params.get('g', 9.81)
                    
                    KE = 0.5 * m * l**2 * theta_dot**2
                    PE = m * g * l * (1 - np.cos(theta))
                    E_total = KE + PE
                    
                elif system_name == 'double_pendulum':
                    theta1, theta1_dot, theta2, theta2_dot = y[0], y[1], y[2], y[3]
                    
                    m1 = params.get('m1', 1.0)
                    m2 = params.get('m2', 1.0)
                    l1 = params.get('l1', 1.0)
                    l2 = params.get('l2', 1.0)
                    g = params.get('g', 9.81)
                    
                    # Simplified energy calculation
                    KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
                    KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2)
                    KE = KE1 + KE2
                    
                    PE1 = -m1 * g * l1 * np.cos(theta1)
                    PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
                    PE = PE1 + PE2
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
            
            # 3D Animation
            if 'pendulum' in system_name:
                st.subheader("🎬 3D Animation")
                
                # Create 3D visualization
                fig_3d = go.Figure()
                
                if system_name == 'simple_pendulum':
                    theta = y[0]
                    l = params.get('l', 1.0)
                    x_traj = l * np.sin(theta)
                    y_traj = -l * np.cos(theta)
                    
                    # Add trajectory
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_traj, y=y_traj, z=np.zeros_like(x_traj),
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Trajectory'
                    ))
                    
                    # Add current position
                    current_frame = len(x_traj) // 2
                    fig_3d.add_trace(go.Scatter3d(
                        x=[0, x_traj[current_frame]],
                        y=[0, y_traj[current_frame]],
                        z=[0, 0],
                        mode='lines+markers',
                        line=dict(color='red', width=5),
                        marker=dict(size=[5, 10], color=['black', 'red']),
                        name='Pendulum'
                    ))
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)", 
                        zaxis_title="Z (m)",
                        aspectmode='cube'
                    ),
                    title="3D Pendulum Trajectory"
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("🌪️ Chaos Analysis & Nonlinear Dynamics")
    
    if not st.session_state.last_solution:
        st.warning("Please run a simulation first.")
    else:
        solution = st.session_state.last_solution
        t = solution['t']
        y = solution['y']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Chaos Indicators")
            
            # Lyapunov exponent calculation (improved)
            def calculate_lyapunov_spectrum(solution, dt_factor=0.1):
                """Calculate Lyapunov spectrum using finite difference method"""
                y = solution['y']
                t = solution['t']
                dt = (t[1] - t[0]) * dt_factor
                
                if len(y.shape) < 2 or y.shape[0] < 2:
                    return [0]
                
                n_vars = y.shape[0]
                n_points = min(len(t), 1000)  # Limit for performance
                
                # Sample every nth point for performance
                skip = max(1, len(t) // n_points)
                y_sampled = y[:, ::skip]
                t_sampled = t[::skip]
                
                lyapunov_values = []
                
                for i in range(min(2, n_vars//2)):  # Check first 2 phase space dimensions
                    pos = y_sampled[2*i]
                    vel = y_sampled[2*i + 1]
                    
                    # Calculate local divergence rate
                    if len(pos) > 10:
                        # Use velocity as proxy for divergence
                        divergence = np.abs(np.diff(vel))
                        if len(divergence) > 0:
                            avg_div = np.mean(divergence[divergence > 1e-10])
                            lyap = np.log(avg_div) if avg_div > 0 else -10
                            lyapunov_values.append(lyap)
                        else:
                            lyapunov_values.append(-10)
                
                return lyapunov_values if lyapunov_values else [0]
            
            lyapunov_spectrum 