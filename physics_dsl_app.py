import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from complete_physics_dsl import *
import io
import base64
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Physics DSL Compiler",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'compiler' not in st.session_state:
    st.session_state.compiler = None
if 'compilation_result' not in st.session_state:
    st.session_state.compilation_result = None
if 'simulation_result' not in st.session_state:
    st.session_state.simulation_result = None

def main():
    st.markdown('<h1 class="main-header">🚀 Physics DSL Compiler & Simulator</h1>', unsafe_allow_html=True)
    
    # Sidebar for examples and parameters
    with st.sidebar:
        st.header("📚 Example Systems")
        
        example_systems = {
            "Simple Pendulum": SIMPLE_PENDULUM_DSL,
            "Double Pendulum": """
\\system{double_pendulum}

\\defvar{theta1}{Angle}{rad}
\\defvar{theta2}{Angle}{rad}
\\defvar{m1}{Mass}{kg}
\\defvar{m2}{Mass}{kg}
\\defvar{l1}{Length}{m}
\\defvar{l2}{Length}{m}
\\defvar{g}{Acceleration}{m/s^2}

\\lagrangian{0.5*m1*l1^2*\\dot{theta1}^2 + 0.5*m2*(l1^2*\\dot{theta1}^2 + l2^2*\\dot{theta2}^2 + 2*l1*l2*\\dot{theta1}*\\dot{theta2}*\\cos{theta1 - theta2}) + m1*g*l1*\\cos{theta1} + m2*g*(l1*\\cos{theta1} + l2*\\cos{theta2})}

\\initial{theta1=1.0, theta1_dot=0, theta2=0.5, theta2_dot=0}

\\solve{euler_lagrange}
\\animate{double_pendulum}
            """,
            "Spring-Mass System": """
\\system{spring_mass}

\\defvar{x}{Position}{m}
\\defvar{m}{Mass}{kg}
\\defvar{k}{Real}{N/m}

\\lagrangian{0.5*m*\\dot{x}^2 - 0.5*k*x^2}

\\initial{x=1.0, x_dot=0}

\\solve{euler_lagrange}
\\animate{spring_mass}
            """
        }
        
        selected_example = st.selectbox("Choose an example:", list(example_systems.keys()))
        
        if st.button("Load Example"):
            st.session_state.dsl_code = example_systems[selected_example]
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Simulation Parameters")
        
        t_max = st.slider("Simulation Time (s)", 1, 20, 10)
        num_points = st.slider("Number of Points", 100, 2000, 1000)
        
        st.session_state.sim_params = {
            't_max': t_max,
            'num_points': num_points
        }

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="step-header">📝 DSL Code Input</h2>', unsafe_allow_html=True)
        
        # Initialize DSL code if not in session state
        if 'dsl_code' not in st.session_state:
            st.session_state.dsl_code = SIMPLE_PENDULUM_DSL
        
        dsl_input = st.text_area(
            "Enter your Physics DSL code:",
            value=st.session_state.dsl_code,
            height=400,
            key="dsl_input"
        )
        
        # Update session state when text changes
        if dsl_input != st.session_state.dsl_code:
            st.session_state.dsl_code = dsl_input
        
        # Compile button
        if st.button("🔧 Compile & Analyze", type="primary"):
            compile_physics_system(dsl_input)
    
    with col2:
        st.markdown('<h2 class="step-header">🔍 Compilation Results</h2>', unsafe_allow_html=True)
        
        if st.session_state.compilation_result:
            display_compilation_results()
        else:
            st.info("👈 Enter DSL code and click 'Compile & Analyze' to see results")

    # Simulation and Visualization Section
    if st.session_state.compilation_result and st.session_state.compilation_result.get('success'):
        st.markdown("---")
        st.markdown('<h2 class="step-header">🎯 Simulation & Visualization</h2>', unsafe_allow_html=True)
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            if st.button("🚀 Run Simulation", type="primary"):
                run_simulation()
        
        with col4:
            if st.session_state.simulation_result:
                st.success(f"✅ Simulation completed with {len(st.session_state.simulation_result['t'])} time points")
        
        # Display simulation results
        if st.session_state.simulation_result and st.session_state.simulation_result.get('success'):
            display_simulation_results()

def compile_physics_system(dsl_code):
    """Compile the physics system from DSL code"""
    
    with st.spinner("Compiling physics system..."):
        try:
            # Create compiler instance
            compiler = PhysicsCompiler()
            
            # Compile the DSL
            result = compiler.compile_dsl(dsl_code)
            
            # Store results in session state
            st.session_state.compiler = compiler
            st.session_state.compilation_result = result
            
            if result['success']:
                st.success("🎉 Compilation successful!")
            else:
                st.error(f"❌ Compilation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Compilation error: {str(e)}")
            st.session_state.compilation_result = {'success': False, 'error': str(e)}

def display_compilation_results():
    """Display the compilation results"""
    
    result = st.session_state.compilation_result
    
    if result['success']:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**✅ Compilation Successful!**")
        st.markdown(f"**System:** {result['system_name']}")
        st.markdown(f"**Coordinates:** {', '.join(result['coordinates'])}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display AST
        with st.expander("🌳 Abstract Syntax Tree"):
            compiler = st.session_state.compiler
            for i, node in enumerate(compiler.ast):
                st.code(f"{i+1}. {node}", language="text")
        
        # Display variables
        with st.expander("📊 System Variables"):
            compiler = st.session_state.compiler
            for var_name, var_info in compiler.variables.items():
                st.write(f"**{var_name}**: {var_info['type']} [{var_info['unit']}]" + 
                        (" [Vector]" if var_info['vector'] else ""))
        
        # Display equations
        with st.expander("⚡ Equations of Motion"):
            for coord, eq in result['equations'].items():
                st.latex(f"{coord} = {sp.latex(eq)}")
        
        # Display parameters
        with st.expander("⚙️ System Parameters"):
            compiler = st.session_state.compiler
            for param, value in compiler.simulator.parameters.items():
                st.write(f"**{param}**: {value}")
        
        # Display initial conditions
        with st.expander("🎯 Initial Conditions"):
            compiler = st.session_state.compiler
            for var, val in compiler.initial_conditions.items():
                st.write(f"**{var}**: {val}")
                
    else:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.markdown(f"**❌ Compilation Failed**")
        st.markdown(f"**Error:** {result.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)

def run_simulation():
    """Run the numerical simulation"""
    
    if not st.session_state.compiler:
        st.error("No compiled system available")
        return
    
    compiler = st.session_state.compiler
    sim_params = st.session_state.sim_params
    
    with st.spinner("Running simulation..."):
        try:
            # Run simulation
            t_span = (0, sim_params['t_max'])
            solution = compiler.simulate(t_span, sim_params['num_points'])
            
            # Store result
            st.session_state.simulation_result = solution
            
            if solution['success']:
                st.success(f"✅ Simulation completed successfully!")
            else:
                st.error(f"❌ Simulation failed: {solution.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            st.session_state.simulation_result = {'success': False, 'error': str(e)}

def display_simulation_results():
    """Display simulation results with plots"""
    
    solution = st.session_state.simulation_result
    compiler = st.session_state.compiler
    
    if not solution['success']:
        st.error(f"Cannot display results: {solution.get('error', 'Simulation failed')}")
        return
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌀 Phase Space", "⚡ Energy Analysis", "🎬 Animation"])
    
    with tab1:
        display_time_series(solution)
    
    with tab2:
        display_phase_space(solution)
    
    with tab3:
        display_energy_analysis(solution, compiler)
    
    with tab4:
        display_animation_controls(solution, compiler)

def display_time_series(solution):
    """Display time series plots"""
    
    st.subheader("📈 Time Series Analysis")
    
    t = solution['t']
    y = solution['y']
    coordinates = solution['coordinates']
    
    # Create subplots for positions and velocities
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Position vs Time', 'Velocity vs Time'],
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot positions
    for i, coord in enumerate(coordinates):
        fig.add_trace(
            go.Scatter(
                x=t, 
                y=y[2*i],
                name=f'{coord}',
                line=dict(color=colors[i % len(colors)]),
                mode='lines'
            ),
            row=1, col=1
        )
    
    # Plot velocities  
    for i, coord in enumerate(coordinates):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y[2*i + 1], 
                name=f'{coord}_dot',
                line=dict(color=colors[i % len(colors)], dash='dash'),
                mode='lines'
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, title="System Dynamics")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data download
    if st.button("📥 Download Data"):
        # Create CSV data
        import pandas as pd
        
        data_dict = {'time': t}
        for i, coord in enumerate(coordinates):
            data_dict[coord] = y[2*i]
            data_dict[f'{coord}_dot'] = y[2*i + 1]
        
        df = pd.DataFrame(data_dict)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{st.session_state.compiler.system_name}_data.csv",
            mime="text/csv"
        )

def display_phase_space(solution):
    """Display phase space plots"""
    
    st.subheader("🌀 Phase Space Analysis")
    
    y = solution['y']
    coordinates = solution['coordinates']
    
    # Select coordinate for phase space
    if len(coordinates) > 1:
        selected_coord = st.selectbox("Select coordinate for phase space:", coordinates)
        coord_idx = coordinates.index(selected_coord)
    else:
        coord_idx = 0
        selected_coord = coordinates[0]
    
    # Plot phase space
    position = y[2 * coord_idx]
    velocity = y[2 * coord_idx + 1]
    
    fig = go.Figure()
    
    # Trajectory
    fig.add_trace(go.Scatter(
        x=position,
        y=velocity,
        mode='lines',
        name='Trajectory',
        line=dict(color='blue', width=2)
    ))
    
    # Start point
    fig.add_trace(go.Scatter(
        x=[position[0]],
        y=[velocity[0]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=10)
    ))
    
    # End point
    fig.add_trace(go.Scatter(
        x=[position[-1]],
        y=[velocity[-1]],
        mode='markers', 
        name='End',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f"Phase Space: {selected_coord}",
        xaxis_title=f"Position {selected_coord}",
        yaxis_title=f"Velocity {selected_coord}_dot",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_energy_analysis(solution, compiler):
    """Display energy analysis"""
    
    st.subheader("⚡ Energy Analysis")
    
    t = solution['t']
    y = solution['y']
    system_name = compiler.system_name
    parameters = compiler.simulator.parameters
    
    try:
        # Calculate energies based on system type
        if 'pendulum' in system_name.lower():
            KE, PE, E_total = calculate_pendulum_energy(y, parameters, system_name)
        elif 'spring' in system_name.lower():
            KE, PE, E_total = calculate_spring_energy(y, parameters)
        else:
            st.warning("Energy calculation not implemented for this system type")
            return
        
        # Plot energy
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Kinetic Energy', 'Potential Energy', 'Total Energy', 'Energy Overview']
        )
        
        # Kinetic Energy
        fig.add_trace(go.Scatter(x=t, y=KE, name='Kinetic', line=dict(color='red')), row=1, col=1)
        
        # Potential Energy
        fig.add_trace(go.Scatter(x=t, y=PE, name='Potential', line=dict(color='blue')), row=1, col=2)
        
        # Total Energy
        fig.add_trace(go.Scatter(x=t, y=E_total, name='Total', line=dict(color='green')), row=2, col=1)
        
        # Overview
        fig.add_trace(go.Scatter(x=t, y=KE, name='Kinetic', line=dict(color='red')), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=PE, name='Potential', line=dict(color='blue')), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=E_total, name='Total', line=dict(color='green', width=3)), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title="Energy Conservation Analysis")
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Energy (J)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Energy conservation check
        energy_variation = (np.max(E_total) - np.min(E_total)) / np.mean(E_total) * 100
        
        if energy_variation < 1:
            st.success(f"✅ Energy well conserved (variation: {energy_variation:.3f}%)")
        elif energy_variation < 5:
            st.warning(f"⚠️ Energy moderately conserved (variation: {energy_variation:.2f}%)")
        else:
            st.error(f"❌ Poor energy conservation (variation: {energy_variation:.1f}%)")
            
    except Exception as e:
        st.error(f"Error calculating energy: {str(e)}")

def calculate_pendulum_energy(y, parameters, system_name):
    """Calculate energy for pendulum systems"""
    
    m = parameters.get('m', parameters.get('m1', 1.0))
    l = parameters.get('l', parameters.get('l1', 1.0))
    g = parameters.get('g', 9.81)
    
    if 'double' in system_name.lower():
        # Double pendulum
        theta1, theta1_dot, theta2, theta2_dot = y[0], y[1], y[2], y[3]
        
        m1 = parameters.get('m1', 1.0)
        m2 = parameters.get('m2', 1.0)
        l1 = parameters.get('l1', 1.0)
        l2 = parameters.get('l2', 1.0)
        
        # Kinetic energy
        KE1 = 0.5 * m1 * l1**2 * theta1_dot**2
        KE2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                          2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
        KE = KE1 + KE2
        
        # Potential energy
        PE1 = -m1 * g * l1 * np.cos(theta1)
        PE2 = -m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
        PE = PE1 + PE2
        
    else:
        # Simple pendulum
        theta = y[0]
        theta_dot = y[1]
        
        # Kinetic energy
        KE = 0.5 * m * l**2 * theta_dot**2
        
        # Potential energy (reference at bottom)
        PE = m * g * l * (1 - np.cos(theta))
    
    E_total = KE + PE
    return KE, PE, E_total

def calculate_spring_energy(y, parameters):
    """Calculate energy for spring-mass system"""
    
    m = parameters.get('m', 1.0)
    k = parameters.get('k', 1.0)
    
    x = y[0]
    x_dot = y[1]
    
    # Kinetic energy
    KE = 0.5 * m * x_dot**2
    
    # Potential energy
    PE = 0.5 * k * x**2
    
    E_total = KE + PE
    return KE, PE, E_total

def display_animation_controls(solution, compiler):
    """Display animation controls"""
    
    st.subheader("🎬 System Animation")
    
    system_name = compiler.system_name.lower()
    
    if 'pendulum' in system_name:
        st.info("🎥 Pendulum animation would be displayed here in a full implementation")
        st.write("Animation features would include:")
        st.write("- Real-time pendulum motion")
        st.write("- Trail visualization")
        st.write("- Speed controls")
        st.write("- Export options")
    else:
        st.info("🎥 Animation for this system type is not yet implemented")
    
    # For now, show a static representation
    if st.button("📊 Show Static Visualization"):
        display_static_system_plot(solution, compiler)

def display_static_system_plot(solution, compiler):
    """Display a static visualization of the system"""
    
    system_name = compiler.system_name.lower()
    t = solution['t']
    y = solution['y']
    parameters = compiler.simulator.parameters
    
    if 'pendulum' in system_name:
        # Plot pendulum positions over time
        fig = go.Figure()
        
        if 'double' in system_name:
            # Double pendulum
            theta1 = y[0]
            theta2 = y[2]
            l1 = parameters.get('l1', 1.0)
            l2 = parameters.get('l2', 1.0)
            
            # Positions
            x1 = l1 * np.sin(theta1)
            y1 = -l1 * np.cos(theta1)
            x2 = x1 + l2 * np.sin(theta2)
            y2 = y1 - l2 * np.cos(theta2)
            
            # Plot trajectories
            fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Bob 1 Trajectory'))
            fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='Bob 2 Trajectory'))
            
            # Plot final configuration
            fig.add_trace(go.Scatter(x=[0, x1[-1], x2[-1]], y=[0, y1[-1], y2[-1]], 
                                   mode='lines+markers', name='Final Configuration'))
        else:
            # Simple pendulum
            theta = y[0]
            l = parameters.get('l', 1.0)
            
            x = l * np.sin(theta)
            y_pos = -l * np.cos(theta)
            
            fig.add_trace(go.Scatter(x=x, y=y_pos, mode='lines', name='Bob Trajectory'))
            
            # Final configuration
            fig.add_trace(go.Scatter(x=[0, x[-1]], y=[0, y_pos[-1]], 
                                   mode='lines+markers', name='Final Configuration'))
        
        fig.update_layout(
            title="System Configuration",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            showlegend=True,
            width=600,
            height=600
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)

# Example DSL for reference
SIMPLE_PENDULUM_DSL = """
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
"""

if __name__ == "__main__":
    main()
