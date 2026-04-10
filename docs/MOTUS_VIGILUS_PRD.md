# 📘 MOTUS-VIGILUS — PRODUCT REQUIREMENTS DOCUMENT

## I. VISION IMPÉRIALE

| Paramètre | Valeur |
|-----------|--------|
| Objectif | Extraire animations depuis vidéo → .fbx R15 compatible Roblox |
| Coût | 0€ (Colab gratuit + open-source) |
| Statut | Forge Extérieure (standalone, non intégrée EXODUS) |

## II. ARCHITECTURE DES 2 FRÉGATES

| Frégate | Script | Input | Output | Stack |
|---------|--------|-------|--------|-------|
| U-ALPHA (L'Auspex) | motus_extract.py | .mp4 | .npz | MediaPipe, NumPy, SciPy, PySceneDetect |
| U-GAMMA (La Forge) | motus_forge.py | .npz + .blend | .fbx | Blender 4.x headless |

### Contrat .npz (Interface U-ALPHA → U-GAMMA)

```python
{
    "rotations": np.float32,       # (N_frames, 15, 4) quaternions WXYZ
    "root_position": np.float32,   # (N_frames, 3) XYZ mètres
    "bone_names": [                # 15 os R15 officiels
        "LowerTorso", "UpperTorso", "Head",
        "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
        "RightUpperLeg", "RightLowerLeg", "RightFoot"
    ],
    "fps": int,
    "duration": float,
    "source_fps": int,
    "person_index": int,
    "total_persons": int
}
```

### Hiérarchie Os R15

```
Root → HumanoidRootNode → LowerTorso (PARENT)
                            ├── UpperTorso
                            │    ├── Head
                            │    ├── LeftUpperArm → LeftLowerArm → LeftHand
                            │    └── RightUpperArm → RightLowerArm → RightHand
                            ├── LeftUpperLeg → LeftLowerLeg → LeftFoot
                            └── RightUpperLeg → RightLowerLeg → RightFoot
```

## III. STACK TECHNIQUE

| Composant | Version | Usage |
|-----------|---------|-------|
| Python | 3.10+ | Orchestration |
| MediaPipe | Latest | Pose Landmarker (pose_world_landmarks 3D) |
| NumPy | Latest | Calcul vectoriel (quaternions) |
| SciPy | Latest | Smoothing (Savitzky-Golay) + Interpolation Bézier |
| PySceneDetect | Latest | Détection de cuts vidéo |
| OpenCV | Latest | Décodage vidéo |
| Blender | 4.0+ | Export FBX headless |
| Google Colab | T4 GPU | Exécution cloud |

## IV. CONTRAINTES

- Étanchéité : U-ALPHA ne connaît pas Blender. U-GAMMA ne connaît pas MediaPipe.
- Durée max : 60 secondes vidéo source.
- Personnages : Max 4 sujets simultanés.
- Axes : X_Roblox = -X_MediaPipe, Y = Y, Z_Roblox = -Z_MediaPipe.

## V. MÉTRIQUES DE SUCCÈS

| Métrique | Cible |
|----------|-------|
| Temps traitement (1 min vidéo) | < 15 min (Colab T4) |
| Qualité animation | Pas de joint popping |
| Code source | < 500 lignes total |
| Coût | 0€ récurrent |
