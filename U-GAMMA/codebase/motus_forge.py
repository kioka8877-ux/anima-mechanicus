#!/usr/bin/env python3
"""MOTUS-VIGILUS — Frégate U-GAMMA : La Forge — v3 FIX FINAL
Conversion .npz → .fbx via Blender headless

DIAGNOSTIC COMPLET :
  U-ALPHA stocke des rotations MONDE : Q_world = vec2quat(rest_vector, observed_bone_dir)
  motus_forge v1/v2 les appliquait comme LOCAL → chaque bone hérite parent + se re-tourne
  → "atomes volants" en cascade

FIX :
  1. root_position → root bone (pas l'objet armature)
  2. Conversion world→local avant application sur chaque pose bone
     local_i = inv(world_parent_i) @ world_i
"""

import bpy
import sys
import numpy as np
from mathutils import Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
if len(argv) < 2:
    print("[ERREUR] Usage: blender -b template.blend -P motus_forge.py -- input.npz output.fbx")
    sys.exit(1)

npz_path = argv[0]
fbx_path = argv[1]

print(f"[U-GAMMA] Chargement : {npz_path}")
data = np.load(npz_path, allow_pickle=True)
rotations     = data["rotations"]
root_position = data["root_position"]
bone_names    = data["bone_names"]
fps           = int(data["fps"])
n_frames      = len(rotations)
print(f"[U-GAMMA] {n_frames} frames @ {fps} FPS — {len(bone_names)} bones")

# ── Hiérarchie parent .npz (R15 sans le bone Root du template) ─────────────
# LowerTorso est la racine de mouvement (son parent Blender = Root, toujours identity)
NPZ_PARENT = {
    'LowerTorso':    None,
    'UpperTorso':    'LowerTorso',
    'Head':          'UpperTorso',
    'LeftUpperArm':  'UpperTorso',
    'LeftLowerArm':  'LeftUpperArm',
    'LeftHand':      'LeftLowerArm',
    'RightUpperArm': 'UpperTorso',
    'RightLowerArm': 'RightUpperArm',
    'RightHand':     'RightLowerArm',
    'LeftUpperLeg':  'LowerTorso',
    'LeftLowerLeg':  'LeftUpperLeg',
    'LeftFoot':      'LeftLowerLeg',
    'RightUpperLeg': 'LowerTorso',
    'RightLowerLeg': 'RightUpperLeg',
    'RightFoot':     'RightLowerLeg',
}


def find_bone(armature, name):
    pose_bones = armature.pose.bones
    if name in pose_bones:
        return pose_bones[name]
    norm = name.lower().replace(" ", "").replace("_", "")
    for pb in pose_bones:
        if pb.name.lower().replace(" ", "").replace("_", "") == norm:
            return pb
    return None


# ── Trouver l'armature ──────────────────────────────────────────────────────
armature = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature = obj
        break

if not armature:
    print("[ERREUR] Aucune armature trouvée dans le template")
    sys.exit(1)

print(f"[U-GAMMA] Armature : {armature.name} ({len(armature.pose.bones)} bones)")

# Armature fixe à l'origine — tout le mouvement passe par les bones
armature.location = Vector((0.0, 0.0, 0.0))
armature.rotation_euler = (0.0, 0.0, 0.0)
armature.scale = (1.0, 1.0, 1.0)

# ── Mapper les bones ────────────────────────────────────────────────────────
bone_map = {}
missing  = []
for bname in bone_names:
    pb = find_bone(armature, str(bname))
    if pb:
        bone_map[str(bname)] = pb
    else:
        missing.append(str(bname))

if missing:
    print(f"[WARN] Bones manquants : {missing}")
print(f"[U-GAMMA] {len(bone_map)}/{len(bone_names)} bones mappés")

# ── Identifier le bone racine pour root_position ───────────────────────────
ROOT_CANDIDATES = ["HumanoidRootNode", "Root", "root", "HumanoidRootPart", "Hips", "hips", "Pelvis", "LowerTorso"]
root_bone = None
for c in ROOT_CANDIDATES:
    root_bone = find_bone(armature, c)
    if root_bone:
        print(f"[U-GAMMA] Root bone : {root_bone.name}")
        break

# ── Paramètres de scène ─────────────────────────────────────────────────────
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = n_frames
bpy.context.scene.render.fps  = fps

bpy.context.view_layer.objects.active = armature
armature.select_set(True)

# ── Boucle principale ───────────────────────────────────────────────────────
bone_name_list = [str(b) for b in bone_names]
bone_idx_map   = {n: i for i, n in enumerate(bone_name_list)}

for frame_idx in range(n_frames):
    bpy.context.scene.frame_set(frame_idx + 1)

    # root_position → root bone (FIX v2)
    if root_bone is not None:
        pos = root_position[frame_idx]
        root_bone.location = Vector((float(pos[0]), float(pos[1]), float(pos[2])))
        root_bone.keyframe_insert(data_path="location", frame=frame_idx + 1)

    # Construire le dict des rotations MONDE pour cette frame
    world_quats = {}
    for bname in bone_name_list:
        idx = bone_idx_map[bname]
        w, x, y, z = rotations[frame_idx, idx]
        world_quats[bname] = Quaternion((float(w), float(x), float(y), float(z)))

    # Appliquer en LOCAL : local_i = inv(world_parent_i) @ world_i  (FIX v3)
    for bname in bone_name_list:
        pb = bone_map.get(bname)
        if pb is None:
            continue

        world_q  = world_quats[bname]
        parent_n = NPZ_PARENT.get(bname)

        if parent_n and parent_n in world_quats:
            parent_q = world_quats[parent_n]
            local_q  = parent_q.inverted() @ world_q
        else:
            # Bone racine : local = world (parent du .npz = Root = identity)
            local_q = world_q

        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = local_q
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

    if frame_idx % 100 == 0:
        print(f"[U-GAMMA] Frame {frame_idx + 1}/{n_frames}")

# ── Export FBX ──────────────────────────────────────────────────────────────
print(f"[U-GAMMA] Export FBX → {fbx_path}")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    object_types={'ARMATURE', 'MESH'},
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_simplify_factor=0.0,
    axis_forward='-Z',
    axis_up='Y',
    add_leaf_bones=False,
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    bake_space_transform=False,
)
print(f"[U-GAMMA] Forge terminée — {fbx_path}")
