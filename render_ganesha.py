import os
import json
import subprocess

def create_blender_script(output_dir, model_path, num_views=32):
    script_content = f"""
import bpy
import os
import json
import math

def setup_camera():
    cam_data = bpy.data.cameras.new("CameraData")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.angle = math.radians(45)
    
    # Target kamera di titik pusat
    empty = bpy.data.objects.new("Target", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    cam_constraint = cam_obj.constraints.new(type='TRACK_TO')
    cam_constraint.target = empty
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    return cam_obj

def render_views(model_path, output_dir, num_views):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    if bpy.context.scene.world is None:
        new_world = bpy.data.worlds.new("NewWorld")
        bpy.context.scene.world = new_world
    
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 3.5
    
    bpy.context.scene.view_settings.exposure = 1.0
    try:
        bpy.context.scene.view_settings.look = 'AgX - High Contrast'
    except:
        bpy.context.scene.view_settings.look = 'High Contrast'
    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    
    # 1. Import Model
    try:
        bpy.ops.import_scene.gltf(filepath=model_path)
    except Exception as e:
        print(f"Gagal import: {{e}}")
        return

    # 2. AUTO-CENTERING: Memastikan objek di tengah (0,0,0)
    # Pilih semua objek mesh yang baru diimport
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
    
    # Gabungkan atau atur origin ke pusat massa, lalu pindahkan ke (0,0,0)
    if bpy.context.selected_objects:
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        for obj in bpy.context.selected_objects:
            obj.location = (0, 0, 0)

    # 3. Bersihkan Alas/Pedestal
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            if any(x in obj.name.lower() for x in ['plane', 'cube', 'pedestal', 'base']):
                bpy.data.objects.remove(obj, do_unlink=True)
    
    cam_obj = setup_camera()
    
    transforms = {{
        "camera_angle_x": cam_obj.data.angle,
        "frames": []
    }}

    # Jarak kamera (Radius)
    radius = 5.8 
    
    for i in range(num_views):
        angle = (i / num_views) * 2 * math.pi
        cam_obj.location = (radius * math.cos(angle), radius * math.sin(angle), 1.8)
        
        bpy.context.view_layer.update()
        
        file_name = f"view_{{i:03d}}.png"
        bpy.context.scene.render.filepath = os.path.join(output_dir, file_name)
        
        bpy.ops.render.render(write_still=True)
        
        matrix_list = [list(row) for row in cam_obj.matrix_world]
        transforms["frames"].append({{
            "file_path": file_name,
            "transform_matrix": matrix_list
        }})
        
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=4)

render_views('{model_path}', '{output_dir}', {num_views})
"""
    with open('temp_blender_script.py', 'w') as f:
        f.write(script_content)

def main():
    base_dir = "/home/pande/TRELLIS.2---Pande/datasets/Ganesha_Dataset"
    folders = [f for f in os.listdir(base_dir) if f.startswith('Ganesha') and os.path.isdir(os.path.join(base_dir, f))]
    folders.sort()
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        glb_path = os.path.join(folder_path, "model.glb")
        
        if os.path.exists(glb_path):
            print(f"🚀 RENDERING: {folder}")
            create_blender_script(folder_path, glb_path)
            subprocess.run(["/usr/bin/blender", "-b", "-P", "temp_blender_script.py"], check=True)
            print(f"✅ SELESAI: {folder}")
            
            # Hapus baris break di bawah ini jika hasil tes sudah presisi di tengah
            #break 

if __name__ == "__main__":
    main()