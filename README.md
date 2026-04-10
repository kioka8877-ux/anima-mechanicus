# ⚔️ MOTUS-VIGILUS

> "Le Mouvement Veille" — Extracteur Cinétique Universel pour Avatars Roblox R15

## Vision

MOTUS-VIGILUS extrait l'animation d'une vidéo `.mp4` et la convertit en fichier `.fbx` compatible avec les avatars Roblox R15. Pipeline 100% gratuit, exécuté sur Google Colab.

## Architecture

| Frégate | Script | Rôle | Input | Output |
|---------|--------|------|-------|--------|
| **U-ALPHA** (L'Auspex) | `U-ALPHA/codebase/motus_extract.py` | Extraction + Transmutation | `.mp4` | `.npz` |
| **U-GAMMA** (La Forge) | `U-GAMMA/codebase/motus_forge.py` | Manifestation FBX | `.npz` + `.blend` | `.fbx` |

## Pipeline

```
📹 .mp4 → [Frégate U-ALPHA] → 📦 .npz → [Frégate U-GAMMA] → 🎮 .fbx
```

## Quick Start (Google Colab)

### Étape 1 — Frégate U-ALPHA
1. Ouvrir `U-ALPHA/MOTUS_VIGILUS_ALPHA.ipynb` dans Google Colab
2. Uploader une vidéo `.mp4`
3. Configurer (FPS, Lissage, Root Motion)
4. Lancer → Télécharger les fichiers `.npz`

### Étape 2 — Frégate U-GAMMA
1. Ouvrir `U-GAMMA/MOTUS_VIGILUS_GAMMA.ipynb` dans Google Colab
2. Uploader les fichiers `.npz` de l'étape 1
3. Lancer → Télécharger `MOTUS_VIGILUS.fbx`

## Spécifications

- **Extraction** : MediaPipe Pose Landmarker (33 landmarks 3D)
- **Multi-personnage** : Jusqu'à 4 sujets (1 fichier `.npz` par personne)
- **Détection de scènes** : PySceneDetect (découpe automatique aux cuts)
- **Lissage** : Savitzky-Golay (scipy)
- **Upscaling temporel** : Interpolation Bézier (30/60/120 FPS)
- **Export** : Blender 4.x headless → FBX avec bake animation
- **Coût** : 0€ (Colab gratuit + open-source)

## Structure

```
MOTUS-VIGILUS/
├── docs/              # Documentation technique (PRD, Roadmap, State)
├── U-ALPHA/           # Frégate U-ALPHA — Extraction (MP4 → NPZ)
│   ├── codebase/      # Script motus_extract.py
│   ├── inputs/        # Vidéos .mp4 sources
│   ├── outputs/       # Fichiers .npz extraits
│   └── *.ipynb        # Notebook Colab U-ALPHA
├── U-GAMMA/           # Frégate U-GAMMA — Forge (NPZ → FBX)
│   ├── codebase/      # Script motus_forge.py
│   ├── inputs/        # Fichiers .npz (depuis U-ALPHA)
│   ├── outputs/       # Fichiers .fbx forgés
│   ├── templates/     # Template Blender R15
│   └── *.ipynb        # Notebook Colab U-GAMMA
├── README.md
└── requirements.txt
```

## Doctrine

Construit selon le framework **ATOM-IC** (Analyse, Transmutation, Optimisation, Manifestation) et les **10 Lois de la Voie Royale**.

## Licence

Usage interne — EXODUS V2 Pipeline.
