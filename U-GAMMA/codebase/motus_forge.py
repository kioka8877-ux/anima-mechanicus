#!/usr/bin/env python3
"""ANIMA-MECHANICUS — Fregate U-GAMMA : La Forge — v4 (rig programmatique)
Conversion .npz → .fbx via Blender headless

Changement v4 : Plus de dependance a r15_template.blend
Le rig R15 est cree programmatiquement (T-pose, 15 os, hierarchie correcte).

Usage:
    blender --background --python motus_forge.py -- input.npz output.fbx
"""

import bpy
import sys
import numpy as np
from mathutils import Quaternion, Vector, Matrix

# ── Arguments ─────────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:]
if len(argv) < 2:
    print("[ERREUR] Usage: blender --background --python motus_forge.py -- input.npz output.fbx")
    sys.exit(1)

npz_path = argv[0]
fbx_path = argv[1]

# ── Chargement .npz ────────────────────────────────────────────────────────────
print(f"[U-GAMMA] Chargement : {npz_path}")
data = np.load(npz_path, allow_pickle=True)
rotations     = data["rotations"]      # (N, 15, 4) quaternions WXYZ
root_position = data["root_position"]  # (N, 3)
bone_names    = [str(b) for b in data["bone_names"]]
fps           = int(data["fps"])
n_frames      = len(rotations)
print(f"[U-GAMMA] {n_frames} frames @ {fps} FPS — {len(bone_names)} bones")

# ── Hierarchie parent (doit correspondre au contrat .npz de U-ALPHA) ───────────
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

# ── Definition du rig R15 (T-pose, LowerTorso a l'origine) ─────────────────────
# Format : (nom, head_xyz, tail_xyz, parent_nom, use_connect)
# LowerTorso tete a (0,0,0) : root_position s'applique directement comme offset
BONE_DEFS = [
    # Colonne vertebrale
    ('LowerTorso',    ( 0.00,  0,  0.00), ( 0.00,  0,  0.20), None,           False),
    ('UpperTorso',    ( 0.00,  0,  0.20), ( 0.00,  0,  0.50), 'LowerTorso',   True ),
    ('Head',          ( 0.00,  0,  0.60), ( 0.00,  0,  0.85), 'UpperTorso',   False),
    # Bras gauche (X positif = gauche personnage)
    ('LeftUpperArm',  ( 0.15,  0,  0.48), ( 0.42,  0,  0.48), 'UpperTorso',   False),
    ('LeftLowerArm',  ( 0.42,  0,  0.48), ( 0.67,  0,  0.48), 'LeftUpperArm', True ),
    ('LeftHand',      ( 0.67,  0,  0.48), ( 0.82,  0,  0.48), 'LeftLowerArm', True ),
    # Bras droit
    ('RightUpperArm', (-0.15,  0,  0.48), (-0.42,  0,  0.48), 'UpperTorso',   False),
    ('RightLowerArm', (-0.42,  0,  0.48), (-0.67,  0,  0.48), 'RightUpperArm',True ),
    ('RightHand',     (-0.67,  0,  0.48), (-0.82,  0,  0.48), 'RightLowerArm',True ),
    # Jambe gauche
    ('LeftUpperLeg',  ( 0.10,  0,  0.00), ( 0.10,  0, -0.40), 'LowerTorso',   False),
    ('LeftLowerLeg',  ( 0.10,  0, -0.40), ( 0.10,  0, -0.80), 'LeftUpperLeg', True ),
    ('LeftFoot',      ( 0.10,  0, -0.80), ( 0.10,  0.12, -0.92), 'LeftLowerLeg',True ),
    # Jambe droite
    ('RightUpperLeg', (-0.10,  0,  0.00), (-0.10,  0, -0.40), 'LowerTorso',   False),
    ('RightLowerLeg', (-0.10,  0, -0.40), (-0.10,  0, -0.80), 'RightUpperLeg',True ),
    ('RightFoot',     (-0.10,  0, -0.80), (-0.10,  0.12, -0.92), 'RightLowerLeg',True ),
]

# ── Nettoyage de la scene par defaut ──────────────────────────────────────────
print("[U-GAMMA] Initialisation scene Blender...")
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ── Creation de l'armature R15 ────────────────────────────────────────────────
print("[U-GAMMA] Creation rig R15...")
arm_data = bpy.data.armatures.new("R15_Armature")
arm_obj  = bpy.data.objects.new("R15", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
arm_obj.select_set(True)

# Entrer en mode EDIT pour creer les bones
bpy.ops.object.mode_set(mode='EDIT')
eb = arm_data.edit_bones

for (name, head, tail, parent_name, use_connect) in BONE_DEFS:
    bone = eb.new(name)
    bone.head = Vector(head)
    bone.tail = Vector(tail)
    bone.use_deform = True
    if parent_name:
        bone.parent = eb[parent_name]
        bone.use_connect = use_connect
    else:
        bone.use_connect = False

bpy.ops.object.mode_set(mode='OBJECT')
print(f"[U-GAMMA] Rig R15 cree : {len(arm_data.bones)} bones")

# ── Parametres de scene ────────────────────────────────────────────────────────
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = n_frames
bpy.context.scene.render.fps  = fps

arm_obj.location = Vector((0.0, 0.0, 0.0))
arm_obj.rotation_euler = (0.0, 0.0, 0.0)
arm_obj.scale = (1.0, 1.0, 1.0)

# ── Index des bones ────────────────────────────────────────────────────────────
bone_idx_map = {name: i for i, name in enumerate(bone_names)}

# ── Boucle principale : application des keyframes ────────────────────────────
print(f"[U-GAMMA] Application des {n_frames} frames...")

for frame_idx in range(n_frames):
    bpy.context.scene.frame_set(frame_idx + 1)

    # Root position → LowerTorso (offset depuis sa position de repos)
    lt = arm_obj.pose.bones.get('LowerTorso')
    if lt is not None:
        pos = root_position[frame_idx]
        lt.location = Vector((float(pos[0]), float(pos[1]), float(pos[2])))
        lt.keyframe_insert(data_path="location", frame=frame_idx + 1)

    # Rotations monde pour cette frame
    world_quats = {}
    for bname in bone_names:
        idx = bone_idx_map.get(bname)
        if idx is None:
            continue
        w, x, y, z = rotations[frame_idx, idx]
        world_quats[bname] = Quaternion((float(w), float(x), float(y), float(z)))

    # Conversion world → local et application (FIX v3 conserve)
    for bname in bone_names:
        pb = arm_obj.pose.bones.get(bname)
        if pb is None:
            continue

        world_q  = world_quats.get(bname)
        if world_q is None:
            continue

        parent_n = NPZ_PARENT.get(bname)
        if parent_n and parent_n in world_quats:
            parent_q = world_quats[parent_n]
            local_q  = parent_q.inverted() @ world_q
        else:
            local_q = world_q

        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = local_q
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx + 1)

    if (frame_idx + 1) % 100 == 0 or frame_idx == n_frames - 1:
        print(f"  Frame {frame_idx + 1}/{n_frames}")

# ── Export FBX ────────────────────────────────────────────────────────────────
print(f"[U-GAMMA] Export FBX → {fbx_path}")
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    object_types={'ARMATURE'},
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
print(f"[U-GAMMA] Forge terminee — {fbx_path}")
