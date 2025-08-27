```mermaid  
graph TD  
    subgraph "Simulation Pipeline"  
        subgraph "User Interaction and Setup"  
            UI[User Input System] -->|Mass, Length, Angles| PE(Physics Engine)  
            UI -->|System Selection| PE  
            UI -->|Interactive Controls| AV(Animation and Visualization)  
        end  

        subgraph "Physics Engine"  
            PE -->|Solve Equations| EqSolver[Equation Solver]  
            PE -->|Manage Constraints| Constrain[Constraint Management]  
            PE -->|Handle Collisions| Collision[Collision Detection]  
            EqSolver -->|Updated Positions| GR(Graphics and Rendering)  
            Constrain -->|Updated Positions| GR  
            Collision -->|Collision Data| GR  
        end  

        subgraph "Graphics and Rendering"  
            GR -->|Object Models (Cylinders, Spheres)| Render[3D Rendering]  
            GR -->|Graphics Framework (Unity, WebGL)| Render  
            GR -->|Realistic Rendering| Render  
        end  

        subgraph "Animation and Visualization"  
            Render --> AV  
            AV -->|Real-Time Animation| Display[Display Simulation]  
            AV -->|Trajectory Visualization| Display  
        end  

        subgraph "Verification and Refinement"  
            PE --> Data[Data Exchange]  
            Data --> Consistency[Consistency Checks]  
            Consistency --> Testing[Testing and Verification]  
            UI -->|User Feedback| Testing  
            Testing -->|Refinements| UI  
        end  
    end  
