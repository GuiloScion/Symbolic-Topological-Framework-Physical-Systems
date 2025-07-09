# Tutorial 1: Getting Started with the DSL

This tutorial will guide you through your first steps in using the Symbolic Topological Framework's Domain-Specific Language (DSL). By the end of this tutorial, you will be able to define a simple physical entity.

## Prerequisites
* A working installation of the Symbolic Topological Framework (e.g., in a Codespace).
* Basic familiarity with command-line operations.

## Step 1: Open Your Workspace

Ensure your Symbolic Topological Framework project is open in VS Code (preferably within a GitHub Codespace for an easy setup).

## Step 2: Create Your First DSL File

Navigate to the `examples/` folder in your project. Create a new file named `first_entity.dsl`.

## Step 3: Define a Simple Entity

Open `first_entity.dsl` and add the following content:

// Define a basic point particle
DEFINE PointParticle {
mass: Mass;
position: Vector3D;
}


* `DEFINE PointParticle`: This keyword is used to declare a new type of physical entity, `PointParticle`.
* `mass: Mass;`: Declares a property named `mass` with the type `Mass` (from the built-in unit system).
* `position: Vector3D;`: Declares a property named `position` with the type `Vector3D`.

## Step 4: (Future) Compile and Run (Conceptual)

*(Note: The actual compilation and running of DSL files will be implemented in `src/dsl` and `src/simulation` in later stages.)*

In a future step, you would run a command like this in your terminal:
```bash
python scripts/run_dsl.py examples/first_entity.dsl

This would compile your DSL definition and allow you to work with the PointParticle in the framework.

Next Steps
In the next tutorial, we'll explore how to instantiate these entities into a system and define relationships between them.

