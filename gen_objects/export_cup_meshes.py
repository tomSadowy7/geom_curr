import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
import pybullet as p

#this script is needed after generating cup meshes and before running train_all.py
#on each mesh. Infinigen loads mugs in .scene blender files, so we must extract the .obj
#from them, VHACD them (turn from convex hull into VHACD which allows handle grasping)
#and also create URDF for physics

#export scene.blend (from Infinigen) to OBJ mesh with materials and textures
def export_obj(cup_dir):
    blend_path = cup_dir / "scene.blend"
    if not blend_path.exists():
        print(f"no scene.blend in {cup_dir}")
        return None

    out_subdir = cup_dir / "genned_assets_obj"

    cmd = [sys.executable, "-m", "infinigen.tools.export",
           "--input_folder", str(cup_dir),
           "--output_folder", str(out_subdir),
           "-f", "obj", "-r", "1024"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED EXPORTING {result.stderr[-400:]}")
        return None

    obj_src  = out_subdir / "export_scene.blend" / "export_scene.obj"
    mtl_src  = out_subdir / "export_scene.blend" / "export_scene.mtl"
    tex_src  = out_subdir / "export_scene.blend" / "textures"

    obj_dst = cup_dir / "cup_visual.obj"
    mtl_dst = cup_dir / "cup_visual.mtl"
    shutil.copy2(obj_src, obj_dst)
    shutil.copy2(mtl_src, mtl_dst)

    text = obj_dst.read_text()
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.lower().startswith("mtllib"):
            lines[i] = "mtllib cup_visual.mtl\n"
            break
    obj_dst.write_text("".join(lines))
    mtl_text = mtl_dst.read_text().splitlines(keepends=True)
    for i, ln in enumerate(mtl_text):
        if ".png" in ln:
            parts = ln.rstrip().split()
            if parts and parts[-1].endswith(".png") and not parts[-1].startswith("textures/"):
                parts[-1] = f"textures/{parts[-1]}"
                mtl_text[i] = " ".join(parts) + "\n"
    mtl_dst.write_text("".join(mtl_text))

    if tex_src.is_dir():
        tex_dst = cup_dir / "textures"
        tex_dst.mkdir(exist_ok=True)
        for png in tex_src.glob("*.png"):
            shutil.copy2(png, tex_dst / png.name)

    shutil.rmtree(out_subdir, ignore_errors=True)

    return obj_dst, mtl_dst


#decompose visual mesh into convex hulls for physics simulation
def run_vhacd(obj_path):
    col_path = obj_path.parent / "cup_collision.obj"
    client = p.connect(p.DIRECT)
    p.vhacd(str(obj_path), str(col_path), str(obj_path.parent / "vhacd_log.txt"),
        alpha=0.04, resolution=100_000, depth=20,
        planeDownsampling=4, convexhullDownsampling=4,
        pca=0, mode=0, maxNumVerticesPerCH=64)
    p.disconnect(client)

    if col_path.exists():
        n = sum(1 for ln in col_path.read_text().splitlines() if ln.startswith("o convex_"))
        return col_path
    
    print("FAILED VHACD EXTRACTION")
    return None


#write URDF file with visual/collision meshes
def write_urdf(cup_dir, mug_params):
    scale = mug_params.get("scale", 0.20)
    depth = mug_params.get("depth", 0.35)
    x_end = mug_params.get("x_end", 0.25)
    radius = x_end * scale
    height = depth * scale
    mass = 0.30

    Iz = 0.5  * mass * radius**2
    Ixy = mass * (3 * radius**2 + height**2) / 12

    collision_block = textwrap.dedent("""
        <collision>
          <geometry>
            <mesh filename="cup_collision.obj" scale="1 1 1"/>
          </geometry>
        </collision>""")

    urdf = textwrap.dedent(f"""<?xml version="1.0"?>
        <robot name="cup">
          <link name="base_link">

            <inertial>
              <mass value="{mass:.4f}"/>
              <inertia ixx="{Ixy:.6f}" ixy="0" ixz="0"
                       iyy="{Ixy:.6f}" iyz="0"
                       izz="{Iz:.6f}"/>
            </inertial>

            <visual>
              <geometry>
                <mesh filename="cup_visual.obj" scale="1 1 1"/>
              </geometry>
            </visual>

            {collision_block.strip()}

          </link>
        </robot>
    """)

    urdf_path = cup_dir / "cup.urdf"
    urdf_path.write_text(textwrap.dedent(urdf))
    return urdf_path


#process all cups: export geometry, VHACD it, and write URDF
def main(args):
    cup_dir_root = Path(args.cup_dir)
    cup_dirs = sorted(cup_dir_root.glob("CupFactory_*/"))

    print(f"Found {len(cup_dirs)} cups")

    for cup_dir in cup_dirs:
        name = cup_dir.name

        params_path = cup_dir / "params.json"
        with open(params_path) as f:
            mug_params = json.load(f)

        export_result = export_obj(cup_dir)
        obj_path, _ = export_result
        col_path = run_vhacd(obj_path)
        urdf_path = write_urdf(cup_dir, mug_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cup_dir", default="experiment_variants")
    main(ap.parse_args())
