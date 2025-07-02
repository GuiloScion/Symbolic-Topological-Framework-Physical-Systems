def run_enhanced_compiler():
    # Enhanced DSL example with mathematical expressions
    dsl = r"""
    \defvar{T}{Real}{kelvin}
    \defvar{k}{Real}{1}
    \defvar{t}{Real}{s}
    \define{ \op{laplace}(T) = k * T^2 + 3.14 * T - 1.0 }
    \define{ \op{heat_flux}(T, k) = -k * T }
    \boundary{T}
    \symmetry{Noether \invariant energy}
    """

    print("=== Enhanced DSL Compiler Demo ===")
    print(f"Source DSL:\n{dsl}")

    try:
        # Tokenization
        tokens = tokenize(dsl)
        print("\n[Tokens]")
        for token in tokens:
            print(f" - {token}")

        # Parsing
        parser = Parser(tokens)
        ast = parser.parse()

        print("\n[Enhanced AST]")
        for stmt in ast:
            print(f" - {stmt}")

        # Type Checking with Dimensional Analysis
        checker = TypeChecker()
        checker.check(ast)

        # IR Compilation
        compiler = IRCompiler()
        compiler.compile(ast)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

# Run the enhanced compiler
if __name__ == "__main__":
    run_enhanced_compiler()