"""
MATLAB Export Generator for Physics DSL
Adds MATLAB code generation capability to PhysicsCompiler
"""

import sympy as sp
from typing import Dict, List

class MATLABExporter:
    """Export DSL to MATLAB validation script"""
    
    def __init__(self, compiler):
        self.compiler = compiler
        self.symbolic = compiler.symbolic
    
    def export_validation_script(self, equations: Dict[str, sp.Expr], filename: str = None):
        """Generate complete MATLAB validation script"""
        
        if filename is None:
            filename = f"validate_{self.compiler.system_name}.m"
        
        # Get system information
        coordinates = self.compiler.get_coordinates()
        parameters = self.compiler.simulator.parameters
        initial_conditions = self.compiler.simulator.initial_conditions
        
        # Build MATLAB script
        matlab_code = self._generate_header(coordinates, parameters, initial_conditions)
        matlab_code += self._generate_ode_function(equations, coordinates, parameters)
        matlab_code += self._generate_energy_function(coordinates, parameters)
        matlab_code += self._generate_utilities()
        matlab_code += self._generate_animation(coordinates, parameters)
        matlab_code += "end\n"
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(matlab_code)
        
        print(f"✓ Generated MATLAB validation script: {filename}")
        print(f"  Run in MATLAB: >> {filename[:-2]}")
        return filename
    
    def _generate_header(self, coordinates: List[str], parameters: dict, initial_conditions: dict):
        """Generate main function header and setup"""
        
        func_name = f"validate_{self.compiler.system_name}"
        
        code = f"""function {func_name}()
    % {self.compiler.system_name.upper().replace('_', ' ')} VALIDATION SCRIPT
    % Auto-generated from Physics DSL
    % Logs data at exact 0.01s intervals for hand calculation comparison
    
    % Parameters
"""
        
        # Write parameters
        for param, value in parameters.items():
            code += f"    {param} = {value:.6f};\n"
        
        code += "\n    % Initial conditions\n"
        
        # Build initial condition vector and print statements
        y0_list = []
        for i, coord in enumerate(coordinates):
            val = initial_conditions.get(coord, 0.0)
            y0_list.append(f"{val:.6f}")
            code += f"    {coord}_0 = {val:.6f};\n"
            
            vel_val = initial_conditions.get(f"{coord}_dot", 0.0)
            y0_list.append(f"{vel_val:.6f}")
            code += f"    {coord}_dot_0 = {vel_val:.6f};\n"
        
        code += f"    y0 = [{'; '.join(y0_list)}];\n"
        
        code += """
    % Time settings - EXACT 0.01s intervals
    dt = 0.01;
    t_end = 10.0;
    tspan = 0:dt:t_end;
    
    % ODE solver options for accuracy
    options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    
    fprintf('========================================\\n');
    fprintf('VALIDATION: {system_name}\\n');
    fprintf('========================================\\n');
""".format(system_name=self.compiler.system_name.upper().replace('_', ' '))
        
        # Print initial conditions
        for coord in coordinates:
            val = initial_conditions.get(coord, 0.0)
            code += f"    fprintf('  {coord}(0) = %.6f\\n', {coord}_0);\n"
        
        code += """
    fprintf('Time step: %.4f s\\n', dt);
    fprintf('Duration: %.2f s\\n', t_end);
    fprintf('========================================\\n\\n');
    
    % Solve ODE
    fprintf('Solving equations of motion...\\n');
    [t, y] = ode45(@(t,y) equations_of_motion(t, y"""
        
        # Add parameters to ODE function call
        for param in parameters.keys():
            code += f", {param}"
        
        code += """), tspan, y0, options);
    fprintf('✓ Solution complete: %d time points\\n\\n', length(t));
    
    % Extract results
"""
        
        # Extract state variables
        for i, coord in enumerate(coordinates):
            code += f"    {coord} = y(:, {2*i + 1});\n"
            code += f"    {coord}_dot = y(:, {2*i + 2});\n"
        
        code += "\n    % Calculate energies\n"
        code += f"    [KE, PE, E_total] = calculate_energy("
        
        energy_args = []
        for coord in coordinates:
            energy_args.extend([coord, f"{coord}_dot"])
        for param in parameters.keys():
            energy_args.append(param)
        
        code += ", ".join(energy_args) + ");\n"
        
        code += """
    
    % Validation checks
    fprintf('========================================\\n');
    fprintf('VALIDATION RESULTS\\n');
    fprintf('========================================\\n');
    
    % Energy conservation
    E_initial = E_total(1);
    E_final = E_total(end);
    energy_drift = abs(E_final - E_initial) / abs(E_initial) * 100;
    max_deviation = max(abs(E_total - E_initial)) / abs(E_initial) * 100;
    
    fprintf('Energy Conservation:\\n');
    fprintf('  Initial Energy: %.8f J\\n', E_initial);
    fprintf('  Final Energy:   %.8f J\\n', E_final);
    fprintf('  Energy Drift:   %.4f%%\\n', energy_drift);
    fprintf('  Max Deviation:  %.4f%%\\n', max_deviation);
    
    if energy_drift < 1.0
        fprintf('  ✓ PASSED: Energy conserved within 1%%\\n');
    else
        fprintf('  ✗ WARNING: Energy drift exceeds 1%%\\n');
    end
    fprintf('========================================\\n\\n');
    
    % Save data
"""
        
        # Build variable list for saving
        save_vars = ['t']
        for coord in coordinates:
            save_vars.extend([coord, f"{coord}_dot"])
        save_vars.extend(['KE', 'PE', 'E_total'])
        
        code += f"    save_data_log({', '.join(save_vars)});\n"
        code += f"    print_comparison_table({', '.join(save_vars)});\n"
        code += f"    create_validation_plots({', '.join(save_vars)});\n"
        
        if self.compiler.system_name in ["simple_pendulum", "double_pendulum"]:
            code += f"    animate_system({', '.join(save_vars[:-3])});\n"
        
        code += "    fprintf('\\n✓ Validation complete!\\n');\n\n"
        
        return code
    
    def _generate_ode_function(self, equations: Dict[str, sp.Expr], coordinates: List[str], parameters: dict):
        """Generate equations of motion function"""
        
        param_list = ", ".join(parameters.keys())
        
        code = f"""    %% Nested function: Equations of Motion
    function dydt = equations_of_motion(~, y, {param_list})
        % Unpack state vector
"""
        
        # Unpack state
        for i, coord in enumerate(coordinates):
            code += f"        {coord} = y({2*i + 1});\n"
            code += f"        {coord}_dot = y({2*i + 2});\n"
        
        code += "\n        dydt = zeros(size(y));\n"
        
        # Generate derivatives
        for i, coord in enumerate(coordinates):
            # Position derivative is velocity
            code += f"        dydt({2*i + 1}) = {coord}_dot;\n"
            
            # Velocity derivative is acceleration
            accel_key = f"{coord}_ddot"
            if accel_key in equations:
                eq = equations[accel_key]
                
                # Substitute all symbols with their MATLAB equivalents
                for j, c in enumerate(coordinates):
                    eq = eq.subs(self.symbolic.get_symbol(c), sp.Symbol(c))
                    eq = eq.subs(self.symbolic.get_symbol(f"{c}_dot"), sp.Symbol(f"{c}_dot"))
                
                # Convert to MATLAB code
                matlab_expr = sp.printing.matlab.matlab_code(eq)
                
                code += f"        dydt({2*i + 2}) = {matlab_expr};\n"
        
        code += "    end\n\n"
        return code
    
    def _generate_energy_function(self, coordinates: List[str], parameters: dict):
        """Generate energy calculation function"""
        
        system = self.compiler.system_name
        
        # Build function signature
        func_args = []
        for coord in coordinates:
            func_args.extend([coord, f"{coord}_dot"])
        for param in parameters.keys():
            func_args.append(param)
        
        code = f"""    %% Nested function: Energy Calculation
    function [KE, PE, E_total] = calculate_energy({", ".join(func_args)})
"""
        
        # System-specific energy calculations
        if system == "simple_pendulum":
            code += """        % Kinetic energy
        KE = 0.5 * m .* l.^2 .* theta_dot.^2;
        
        % Potential energy (reference at lowest point)
        PE = m .* g .* l .* (1 - cos(theta));
"""
        
        elif system == "double_pendulum":
            code += """        % Kinetic energy
        KE1 = 0.5 * m1 * l1^2 .* theta1_dot.^2;
        KE2 = 0.5 * m2 * (l1^2 .* theta1_dot.^2 + l2^2 .* theta2_dot.^2 + ...
                          2 * l1 * l2 .* theta1_dot .* theta2_dot .* cos(theta1 - theta2));
        KE = KE1 + KE2;
        
        % Potential energy
        PE1 = -m1 * g * l1 * cos(theta1);
        PE2 = -m2 * g * (l1 * cos(theta1) + l2 * cos(theta2));
        PE = PE1 + PE2;
"""
        else:
            # Generic energy calculation
            code += """        % Generic energy calculation
        % TODO: Implement system-specific energy
        KE = zeros(size(t));
        PE = zeros(size(t));
"""
        
        code += """        
        % Total energy
        E_total = KE + PE;
    end

"""
        return code
    
    def _generate_utilities(self):
        """Generate utility functions for data logging and plotting"""
        
        coordinates = self.compiler.get_coordinates()
        
        # Build variable names for headers
        var_names = []
        for coord in coordinates:
            var_names.extend([f"{coord}(rad)", f"{coord}_dot(rad/s)"])
        
        code = f"""    %% Nested function: Save Data Log
    function save_data_log(t, """
        
        # Add all variables
        for coord in coordinates:
            code += f"{coord}, {coord}_dot, "
        code += "KE, PE, E_total)\n"
        
        code += f"""        filename = '{self.compiler.system_name}_data.csv';
        
        % Create table
        data_table = table(t"""
        
        for coord in coordinates:
            code += f", {coord}, {coord}_dot"
        code += """, KE, PE, E_total, ...
                          'VariableNames', {'Time_s'"""
        
        for coord in coordinates:
            code += f", '{coord}_rad', '{coord}_dot_rad_s'"
        code += """, 'KE_J', 'PE_J', 'E_total_J'});
        writetable(data_table, filename);
        fprintf('✓ Data saved to: %s\\n', filename);
    end

    %% Nested function: Print Comparison Table
    function print_comparison_table(t, """
        
        for coord in coordinates:
            code += f"{coord}, {coord}_dot, "
        code += "KE, PE, E_total)\n"
        
        code += """        fprintf('\\n========================================\\n');
        fprintf('COMPARISON TABLE (First 0.1 seconds)\\n');
        fprintf('========================================\\n');
        fprintf('%6s"""
        
        for coord in coordinates:
            code += f" | %10s | %10s"
        code += """ | %10s\\n', 't(s)'"""
        
        for coord in coordinates:
            code += f", '{coord}(rad)', 'ω_{coord}(r/s)'"
        code += ", 'E(J)');\n"
        
        code += """        fprintf('-------|"""
        code += "------------|" * (2 * len(coordinates) + 1)
        code += """\\n');
        
        for i = 1:min(11, length(t))
            fprintf('%6.2f"""
        
        for _ in range(2 * len(coordinates) + 1):
            code += " | %10.6f"
        code += "\\n', t(i)"
        
        for coord in coordinates:
            code += f", {coord}(i), {coord}_dot(i)"
        code += ", E_total(i));\n"
        
        code += """        end
        fprintf('========================================\\n');
    end

    %% Nested function: Create Validation Plots
    function create_validation_plots(t, """
        
        for coord in coordinates:
            code += f"{coord}, {coord}_dot, "
        code += "KE, PE, E_total)\n"
        
        num_plots = 2 + len(coordinates) * 2  # angles, velocities, energies, phase spaces
        
        code += f"""        figure('Position', [100, 100, 1200, 800]);
        
        % Angles vs time
        subplot(2, 3, 1);
"""
        
        colors = ['r', 'b', 'g', 'm', 'c']
        for i, coord in enumerate(coordinates):
            color = colors[i % len(colors)]
            if i == 0:
                code += f"        plot(t, {coord}, '{color}-', 'LineWidth', 1.5);"
            else:
                code += f" hold on;\n        plot(t, {coord}, '{color}-', 'LineWidth', 1.5);"
        
        code += """
        xlabel('Time (s)'); ylabel('Angle (rad)');
        title('Angles vs Time');
        legend("""
        
        legend_items = ", ".join([f"'\\{coord}'" for coord in coordinates])
        code += legend_items + """, 'Location', 'best');
        grid on;
        
        % Energy vs time
        subplot(2, 3, 2);
        plot(t, E_total, 'g-', 'LineWidth', 2);
        xlabel('Time (s)'); ylabel('Total Energy (J)');
        title('Total Energy Conservation');
        grid on;
        
        % Energy components
        subplot(2, 3, 3);
        plot(t, KE, 'r-', 'LineWidth', 1.5); hold on;
        plot(t, PE, 'b-', 'LineWidth', 1.5);
        plot(t, E_total, 'g-', 'LineWidth', 2);
        xlabel('Time (s)'); ylabel('Energy (J)');
        title('Energy Components');
        legend('Kinetic', 'Potential', 'Total');
        grid on;
        
        % Phase spaces
"""
        
        for i, coord in enumerate(coordinates):
            code += f"""        subplot(2, 3, {4 + i});
        plot({coord}, {coord}_dot, '{colors[i % len(colors)]}-', 'LineWidth', 1);
        xlabel('\\{coord} (rad)'); ylabel('\\omega_{coord} (rad/s)');
        title('Phase Space - {coord}');
        grid on;
        
"""
        
        code += f"""        sgtitle('{self.compiler.system_name.replace('_', ' ').title()} Validation', 'FontSize', 14);
        saveas(gcf, '{self.compiler.system_name}_validation.png');
        fprintf('✓ Plots saved\\n');
    end

"""
        return code
    
    def _generate_animation(self, coordinates: List[str], parameters: dict):
        """Generate animation function"""
        
        system = self.compiler.system_name
        
        code = f"""    %% Nested function: Animation
    function animate_system(t, """
        
        for coord in coordinates:
            code += f"{coord}, {coord}_dot, "
        code = code.rstrip(", ") + ")\n"
        
        if system == "simple_pendulum":
            code += """        l = 1.0;
        x = l * sin(theta);
        y_pos = -l * cos(theta);
        
        figure('Position', [100, 100, 800, 800]);
        
        for i = 1:5:length(t)
            clf; hold on;
            
            % Trail
            trail_len = min(i, 100);
            plot(x(i-trail_len+1:i), y_pos(i-trail_len+1:i), 'b-', 'LineWidth', 0.5);
            
            % Pendulum
            plot([0 x(i)], [0 y_pos(i)], 'k-', 'LineWidth', 3);
            plot(x(i), y_pos(i), 'ro', 'MarkerSize', 20, 'MarkerFaceColor', 'r');
            plot(0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
            
            axis equal;
            xlim([-l*1.2 l*1.2]); ylim([-l*1.2 l*0.2]);
            title(sprintf('t = %.2f s', t(i)));
            grid on;
            drawnow;
        end
        fprintf('✓ Animation complete\\n');
"""
        
        elif system == "double_pendulum":
            code += """        l1 = 1.0; l2 = 1.0;
        x1 = l1 * sin(theta1);
        y1 = -l1 * cos(theta1);
        x2 = x1 + l2 * sin(theta2);
        y2 = y1 - l2 * cos(theta2);
        
        figure('Position', [100, 100, 800, 800]);
        
        for i = 1:5:length(t)
            clf; hold on;
            
            % Trail of second bob
            trail_len = min(i, 200);
            plot(x2(i-trail_len+1:i), y2(i-trail_len+1:i), 'b-', 'LineWidth', 0.5);
            
            % Pendulum arms
            plot([0 x1(i)], [0 y1(i)], 'k-', 'LineWidth', 3);
            plot([x1(i) x2(i)], [y1(i) y2(i)], 'k-', 'LineWidth', 3);
            
            % Bobs
            plot(0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
            plot(x1(i), y1(i), 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
            plot(x2(i), y2(i), 'bo', 'MarkerSize', 15, 'MarkerFaceColor', 'b');
            
            axis equal;
            max_reach = l1 + l2;
            xlim([-max_reach*1.1 max_reach*1.1]);
            ylim([-max_reach*1.1 max_reach*0.2]);
            title(sprintf('Double Pendulum | t = %.2f s', t(i)));
            grid on;
            drawnow;
        end
        fprintf('✓ Animation complete\\n');
"""
        else:
            code += """        fprintf('Animation not implemented for this system\\n');
"""
        
        code += "    end\n\n"
        return code


# Add method to PhysicsCompiler class
def add_matlab_export(PhysicsCompiler):
    """Add MATLAB export capability to PhysicsCompiler"""
    
    def export_to_matlab(self, equations: Dict[str, sp.Expr] = None, filename: str = None):
        """Export system to MATLAB validation script"""
        if equations is None:
            equations = self.derive_equations()
        
        exporter = MATLABExporter(self)
        return exporter.export_validation_script(equations, filename)
    
    PhysicsCompiler.export_to_matlab = export_to_matlab


# Usage example
if __name__ == "__main__":
    # Import your existing PhysicsCompiler
    from complete_physics_dsl import PhysicsCompiler
    
    # Add MATLAB export capability
    add_matlab_export(PhysicsCompiler)
    
    # Double pendulum DSL
    DOUBLE_PENDULUM_DSL = """
\\system{double_pendulum}

\\defvar{theta1}{Angle}{rad}
\\defvar{theta2}{Angle}{rad}
\\defvar{m1}{Mass}{kg}
\\defvar{m2}{Mass}{kg}
\\defvar{l1}{Length}{m}
\\defvar{l2}{Length}{m}
\\defvar{g}{Acceleration}{m/s^2}

\\lagrangian{0.5*(m1+m2)*l1^2*\\dot{theta1}^2 + 0.5*m2*l2^2*\\dot{theta2}^2 + m2*l1*l2*\\dot{theta1}*\\dot{theta2}*\\cos{theta1-theta2} - (m1+m2)*g*l1*\\cos{theta1} - m2*g*l2*\\cos{theta2}}

\\initial{theta1=1.5708, theta2=1.5708, theta1_dot=0, theta2_dot=0}
\\solve{euler_lagrange}
"""
    
    # Compile DSL
    compiler = PhysicsCompiler()
    result = compiler.compile_dsl(DOUBLE_PENDULUM_DSL)
    
    if result['success']:
        print("✓ DSL compiled successfully")
        
        # Export to MATLAB
        matlab_file = compiler.export_to_matlab(result['equations'])
        
        print(f"\n{'='*60}")
        print("NEXT STEPS:")
        print(f"{'='*60}")
        print(f"1. Open MATLAB")
        print(f"2. Navigate to this directory")
        print(f"3. Run: >> {matlab_file[:-2]}")
        print(f"4. Compare the CSV output with your hand calculations")
        print(f"{'='*60}")
    else:
        print(f"✗ Compilation failed: {result.get('error', 'Unknown error')}")
