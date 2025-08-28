import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# First, let's test if the basic imports work
try:
    from complete_physics_dsl import *
    import_success = True
except Exception as e:
    import_success = False
    import_error = str(e)

# Configure Streamlit page
st.set_page_config(
    page_title="Physics DSL Compiler - Debug",
    page_icon="🚀",
    layout="wide"
)

def main():
    st.title("🚀 Physics DSL Compiler - Debug Mode")
    
    # Check import status first
    if not import_success:
        st.error(f"❌ Import Error: {import_error}")
        st.write("**Troubleshooting Steps:**")
        st.write("1. Make sure `complete_physics_dsl.py` is in the same directory as `app.py`")
        st.write("2. Check that all required packages are installed")
        st.write("3. Verify there are no syntax errors in `complete_physics_dsl.py`")
        
        # Show what we can import
        st.subheader("Available Imports:")
        try:
            import sympy as sp
            st.success("✅ sympy imported successfully")
        except:
            st.error("❌ sympy import failed")
            
        try:
            from scipy.integrate import solve_ivp
            st.success("✅ scipy imported successfully")
        except:
            st.error("❌ scipy import failed")
            
        return
    
    st.success("✅ All imports successful!")
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Simple step-by-step debug
    st.sidebar.header("Debug Steps")
    debug_step = st.sidebar.radio("Select debug step:", [
        "1. Basic Functionality", 
        "2. Tokenization Test", 
        "3. Parsing Test",
        "4. Full Compilation Test"
    ])
    
    if debug_step == "1. Basic Functionality":
        test_basic_functionality()
    elif debug_step == "2. Tokenization Test":
        test_tokenization()
    elif debug_step == "3. Parsing Test":
        test_parsing()
    elif debug_step == "4. Full Compilation Test":
        test_full_compilation()

def test_basic_functionality():
    st.header("1. Basic Functionality Test")
    
    # Test simple operations
    st.write("Testing basic operations...")
    
    try:
        # Test numpy
        arr = np.array([1, 2, 3, 4, 5])
        st.success(f"✅ NumPy working: {arr}")
    except Exception as e:
        st.error(f"❌ NumPy error: {e}")
    
    try:
        # Test sympy
        import sympy as sp
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        st.success(f"✅ SymPy working: {expr}")
    except Exception as e:
        st.error(f"❌ SymPy error: {e}")
    
    try:
        # Test plotly
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        fig.update_layout(title="Test Plot")
        st.plotly_chart(fig)
        st.success("✅ Plotly working")
    except Exception as e:
        st.error(f"❌ Plotly error: {e}")

def test_tokenization():
    st.header("2. Tokenization Test")
    
    # Simple DSL code for testing
    test_code = st.text_area("Test DSL Code:", value="""
\\system{test_system}
\\defvar{x}{Position}{m}
\\defvar{m}{Mass}{kg}
""", height=150)
    
    if st.button("Test Tokenization"):
        try:
            with st.spinner("Tokenizing..."):
                tokens = tokenize(test_code)
                st.success(f"✅ Tokenization successful! Found {len(tokens)} tokens")
                
                # Display tokens
                st.subheader("Tokens:")
                for i, token in enumerate(tokens):
                    st.write(f"{i+1}. {token}")
                    
        except Exception as e:
            st.error(f"❌ Tokenization failed: {e}")
            st.exception(e)

def test_parsing():
    st.header("3. Parsing Test")
    
    test_code = st.text_area("Test DSL Code:", value="""
\\system{simple_test}
\\defvar{x}{Position}{m}
\\defvar{v}{Velocity}{m/s}
\\lagrangian{0.5 * m * v^2}
""", height=150)
    
    if st.button("Test Parsing"):
        try:
            with st.spinner("Parsing..."):
                # Tokenize first
                tokens = tokenize(test_code)
                st.write(f"Tokenized: {len(tokens)} tokens")
                
                # Parse
                parser = MechanicsParser(tokens)
                ast = parser.parse()
                st.success(f"✅ Parsing successful! Generated {len(ast)} AST nodes")
                
                # Display AST
                st.subheader("AST Nodes:")
                for i, node in enumerate(ast):
                    st.write(f"{i+1}. {node}")
                    
        except Exception as e:
            st.error(f"❌ Parsing failed: {e}")
            st.exception(e)

def test_full_compilation():
    st.header("4. Full Compilation Test")
    
    # Use the simplest possible system
    simple_system = """
\\system{minimal_test}

\\defvar{x}{Position}{m}
\\defvar{m}{Mass}{kg}
\\defvar{k}{Real}{N/m}

\\lagrangian{0.5*m*\\dot{x}^2 - 0.5*k*x^2}

\\initial{x=1.0, x_dot=0}

\\solve{euler_lagrange}
"""
    
    test_code = st.text_area("Full System DSL Code:", value=simple_system, height=200)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Full Compilation"):
            try:
                with st.spinner("Compiling..."):
                    compiler = PhysicsCompiler()
                    result = compiler.compile_dsl(test_code)
                    
                    if result['success']:
                        st.success("✅ Compilation successful!")
                        st.write(f"System: {result['system_name']}")
                        st.write(f"Coordinates: {result['coordinates']}")
                        st.session_state.test_compiler = compiler
                        st.session_state.test_result = result
                    else:
                        st.error(f"❌ Compilation failed: {result.get('error', 'Unknown error')}")
                        
            except Exception as e:
                st.error(f"❌ Compilation exception: {e}")
                st.exception(e)
    
    with col2:
        if st.button("Test Simulation") and 'test_compiler' in st.session_state:
            try:
                with st.spinner("Simulating..."):
                    compiler = st.session_state.test_compiler
                    solution = compiler.simulate((0, 5), 500)
                    
                    if solution['success']:
                        st.success("✅ Simulation successful!")
                        
                        # Simple plot
                        t = solution['t']
                        x = solution['y'][0]  # First coordinate
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=t, y=x, name='Position'))
                        fig.update_layout(title="Position vs Time")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"❌ Simulation failed: {solution.get('error', 'Unknown error')}")
                        
            except Exception as e:
                st.error(f"❌ Simulation exception: {e}")
                st.exception(e)

# Error handling wrapper
def safe_main():
    try:
        main()
    except Exception as e:
        st.error("🚨 **Application Error**")
        st.error(f"Error: {str(e)}")
        st.subheader("Debugging Information:")
        st.exception(e)
        
        st.subheader("Troubleshooting:")
        st.write("1. Check that all files are in the correct directory")
        st.write("2. Verify all dependencies are installed: `pip install -r requirements.txt`")
        st.write("3. Restart the Streamlit server: `streamlit run app.py`")
        st.write("4. Check the terminal for additional error messages")

if __name__ == "__main__":
    safe_main()
