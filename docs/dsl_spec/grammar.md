# DSL Grammar

This document outlines the grammar for the Symbolic Topological Framework's Domain-Specific Language.

## Syntax Overview

The DSL is designed to be declarative, allowing users to describe physical systems, their components, and the relationships between them using a high-level syntax.

### Basic Structure

A typical DSL program will define entities, properties, relations, and processes.

// Example DSL Snippet (pseudocode-like)

DEFINE Particle {
mass: Mass;
charge: Charge;
position: Vector3D;
}

SYSTEM MySystem {
particle P1 : Particle { mass=1kg, charge=1C, position=(0,0,0) }
particle P2 : Particle { mass=2kg, charge=0C, position=(1,0,0) }

  RELATION Force(P1, P2) -> Vector3D;

  PROCESS Gravity(P1, P2) {
      APPLY Force;
      CONSTRAINT P1.mass > 0;
  }
}


Formal Grammar (EBNF/BNF - detailed here later)

```ebnf
// Placeholder for EBNF or ANTLR grammar

program = { statement } ;
statement = define_statement | system_statement | process_statement ;

define_statement = "DEFINE" Identifier "{" { property_definition } "}" ;
property_definition = Identifier ":" Type ";" ;

system_statement = "SYSTEM" Identifier "{" { entity_instantiation | relation_definition } "}" ;
entity_instantiation = "particle" Identifier ":" Type "{" { assignment } "}" ;

//... much more detail to be added here

Unit System
The DSL supports a strong unit system, ensuring dimensional consistency throughout the symbolic computations.

Mass: kg

Length: m

Time: s

... (more details on units and conversions)

