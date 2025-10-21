"""
Meshroom Photogrammetry Pipeline
Automates 3D model generation from photo archives
"""

import os
import sys
import zipfile
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
import argparse

class MeshroomPipeline:
    def __init__(self, meshroom_path, output_dir="output"):
        """
        Initialize pipeline
        
        Args:
            meshroom_path: Path to Meshroom installation (e.g., C:/Program Files/Meshroom-2021.1.0)
            output_dir: Base directory for all outputs
        """
        self.meshroom_batch = Path(meshroom_path) / "meshroom_batch.exe"
        self.output_base = Path(output_dir)
        self.output_base.mkdir(exist_ok=True)
        
        if not self.meshroom_batch.exists():
            raise FileNotFoundError(f"Meshroom batch not found at {self.meshroom_batch}")
        
        print(f"[INFO] Pipeline initialized")
        print(f"[INFO] Meshroom: {self.meshroom_batch}")
        print(f"[INFO] Output directory: {self.output_base.absolute()}")
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file to temporary directory"""
        print(f"\n[STEP 1/5] Extracting images from {Path(zip_path).name}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Filter for image files only
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            image_files = [f for f in zip_ref.namelist() 
                          if Path(f).suffix.lower() in image_extensions]
            
            if not image_files:
                raise ValueError("No valid image files found in zip")
            
            print(f"[INFO] Found {len(image_files)} images")
            
            for file in image_files:
                zip_ref.extract(file, extract_to)
            
            # Move all images to root of extract directory (flatten structure)
            for root, dirs, files in os.walk(extract_to):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        src = Path(root) / file
                        dst = extract_to / file
                        if src != dst:
                            shutil.move(str(src), str(dst))
            
            # Clean up empty directories
            for root, dirs, files in os.walk(extract_to, topdown=False):
                for dir in dirs:
                    dir_path = Path(root) / dir
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
        
        return len(image_files)
    
    def validate_images(self, image_dir):
        """Basic image validation"""
        print("\n[STEP 2/5] Validating images")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        images = [f for f in image_dir.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        if len(images) < 3:
            raise ValueError(f"Insufficient images: need at least 3, found {len(images)}")
        
        print(f"[INFO] Validated {len(images)} images")
        return images
    
    def run_meshroom(self, image_dir, project_output):
        """Execute Meshroom batch processing"""
        print("\n[STEP 3/5] Running Meshroom reconstruction")
        print("[INFO] This may take 10-60+ minutes depending on image count and hardware")
        
        cmd = [
            str(self.meshroom_batch),
            "--input", str(image_dir),
            "--output", str(project_output)
        ]
        
        print(f"[CMD] {' '.join(cmd)}")
        
        try:
            # Run with output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(f"[MESHROOM] {line}")
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            
            print("[INFO] Meshroom reconstruction complete")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Meshroom failed with return code {e.returncode}")
    
    def find_mesh_file(self, project_output):
        """Locate the generated mesh file"""
        # Meshroom typically outputs to MeshroomCache/Texturing/
        texturing_dir = project_output / "MeshroomCache" / "Texturing"
        
        if not texturing_dir.exists():
            # Try alternative structure
            for cache_dir in project_output.rglob("Texturing"):
                texturing_dir = cache_dir
                break
        
        if texturing_dir.exists():
            obj_files = list(texturing_dir.rglob("*.obj"))
            if obj_files:
                return obj_files[0]
        
        # Fallback: search entire output
        obj_files = list(project_output.rglob("*.obj"))
        if obj_files:
            return obj_files[0]
        
        return None
    
    def convert_to_stl(self, obj_path, stl_path):
        """Convert OBJ to STL using simple conversion"""
        print("\n[STEP 4/5] Converting to STL format")
        
        try:
            import trimesh
            mesh = trimesh.load(obj_path)
            mesh.export(stl_path)
            print(f"[INFO] STL exported: {stl_path.name}")
        except ImportError:
            print("[WARNING] trimesh not installed, skipping STL conversion")
            print("[INFO] Install with: pip install trimesh")
            return False
        except Exception as e:
            print(f"[WARNING] STL conversion failed: {e}")
            return False
        
        return True
    
    def copy_final_outputs(self, project_output, final_dir, project_name):
        """Copy OBJ and STL to final output directory"""
        print("\n[STEP 5/5] Organizing final outputs")
        
        final_dir.mkdir(exist_ok=True)
        
        # Find and copy OBJ
        obj_file = self.find_mesh_file(project_output)
        if not obj_file:
            raise FileNotFoundError("Could not locate generated mesh file")
        
        final_obj = final_dir / f"{project_name}.obj"
        shutil.copy2(obj_file, final_obj)
        print(f"[INFO] OBJ exported: {final_obj}")
        
        # Copy associated MTL and textures if they exist
        mtl_file = obj_file.with_suffix('.mtl')
        if mtl_file.exists():
            final_mtl = final_dir / f"{project_name}.mtl"
            shutil.copy2(mtl_file, final_mtl)
            
            # Update MTL reference in OBJ
            obj_content = final_obj.read_text()
            obj_content = obj_content.replace(mtl_file.name, final_mtl.name)
            final_obj.write_text(obj_content)
        
        # Copy textures
        texture_dir = obj_file.parent
        for texture in texture_dir.glob("*.png"):
            shutil.copy2(texture, final_dir / texture.name)
        for texture in texture_dir.glob("*.jpg"):
            shutil.copy2(texture, final_dir / texture.name)
        
        # Convert to STL
        final_stl = final_dir / f"{project_name}.stl"
        self.convert_to_stl(final_obj, final_stl)
        
        return final_obj, final_stl
    
    def process(self, zip_path):
        """
        Main pipeline execution
        
        Args:
            zip_path: Path to zip file containing images
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"{zip_path.stem}_{timestamp}"
        project_dir = self.output_base / project_name
        project_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"MESHROOM PIPELINE - Project: {project_name}")
        print(f"{'='*60}")
        
        # Temporary directories
        temp_images = project_dir / "temp_images"
        temp_output = project_dir / "temp_meshroom"
        final_output = project_dir / "final"
        
        temp_images.mkdir(exist_ok=True)
        temp_output.mkdir(exist_ok=True)
        
        try:
            # Extract images
            num_images = self.extract_zip(zip_path, temp_images)
            
            # Validate
            self.validate_images(temp_images)
            
            # Run Meshroom
            self.run_meshroom(temp_images, temp_output)
            
            # Copy final outputs
            obj_file, stl_file = self.copy_final_outputs(temp_output, final_output, project_name)
            
            print(f"\n{'='*60}")
            print("PIPELINE COMPLETE!")
            print(f"{'='*60}")
            print(f"Project directory: {project_dir.absolute()}")
            print(f"Final outputs: {final_output.absolute()}")
            print(f"  - {obj_file.name}")
            if stl_file.exists():
                print(f"  - {stl_file.name}")
            print(f"{'='*60}\n")
            
            # Cleanup temp directories
            print("[INFO] Cleaning up temporary files...")
            shutil.rmtree(temp_images)
            shutil.rmtree(temp_output)
            
            return final_output
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Meshroom Photogrammetry Pipeline - Convert photos to 3D models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python meshroom_pipeline.py -z photos.zip -m "C:/Program Files/Meshroom-2021.1.0"
  python meshroom_pipeline.py -z scan_data.zip -m "C:/Meshroom" -o my_models
        """
    )
    
    parser.add_argument('-z', '--zip', required=True,
                       help='Path to zip file containing images')
    parser.add_argument('-m', '--meshroom', required=True,
                       help='Path to Meshroom installation directory')
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    try:
        pipeline = MeshroomPipeline(args.meshroom, args.output)
        pipeline.process(args.zip)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
