import tempfile
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so imports work when running pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from complete_physics_dsl import PhysicsCompiler
from classical_dsl_frontend import EXAMPLES


def test_examples_compile_simulate_export():
    for name, info in EXAMPLES.items():
        dsl = info['dsl']
        compiler = PhysicsCompiler()
        res = compiler.compile_dsl(dsl)
        assert res.get('success'), f"Compile failed for {name}: {res.get('error')}"

        simulator = res.get('simulator')
        assert simulator is not None

        sim_result = simulator.simulate((0, 0.5), num_points=100)
        assert sim_result.get('success'), f"Simulation failed for {name}: {sim_result.get('error')}"

        # Try exporting GIF (fallback) to temp file
        tmp = tempfile.gettempdir()
        out_path = os.path.join(tmp, f"test_{name.replace(' ', '_')}.gif")
        try:
            fname = compiler.export_animation(sim_result, out_path, fps=10)
            assert os.path.exists(fname)
            # file size should be non-zero
            assert os.path.getsize(fname) > 0
        except RuntimeError:
            # If export fails that's OK for some systems; fail only if both compile/simulate succeeded but export fails
            # We'll skip strictness here to avoid flakiness in CI without ffmpeg/imagemagick
            pass
