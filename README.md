# ANIMA-MECHANICUS

> "L'Âme du Mouvement, Forgée par la Machine" — Pipeline de Motion Capture Vidéo → FBX Roblox R15

Inspiré du lore Warhammer 40 000 — les Prêtres-Techniciens du Dieu-Machine qui insufflent la vie dans les automates.

---

## Vision

**ANIMA-MECHANICUS** extrait les mouvements d'une vidéo de **vrai humain** et les convertit en fichier `.fbx` compatible avec les avatars Roblox R15.

Pipeline 100% gratuit, exécuté sur Google Colab.

Anciennement connu sous le nom de **MOTUS-VIGILUS** (MediaPipe). Cette version remplace le moteur d'extraction par une chaîne académique de pointe : **Gemini 2.0 Flash** (analyse intelligente) + **WHAM** (estimation de pose SMPL).

---

## Architecture — Les 2 Frégates

| Frégate | Script | Rôle | Input | Output |
|---------|--------|------|-------|--------|
| **U-ALPHA** (L'Auspex Cogitateur) | `U-ALPHA/codebase/motus_extract.py` | Analyse Gemini + Extraction WHAM + Transmutation SMPL→R15 | `.mp4` | `.npz` |
| **U-GAMMA** (La Forge) | `U-GAMMA/codebase/motus_forge.py` | Manifestation FBX via Blender headless | `.npz` + `.blend` | `.fbx` |

---

## Pipeline Complet

```
Vidéo .mp4 (humain réel)
        |
        v
[GEMINI 2.0 FLASH]  — L'Oeil Omniscient
  • Analyse la vidéo comme un observateur humain
  • Compte les personnes présentes
  • Identifie les segments exploitables (corps visible, pas de flou)
  • Détecte les coupures de scènes
  • Retourne un rapport JSON : segments, qualité, nb_personnes
        |
        v  (segments validés uniquement)
[WHAM]  — Le Détecteur de Chair
  • Moteur de pose SMPL de niveau académique (CVPR 2024)
  • Reconstruit 24 joints en coordonnées MONDE (pas caméra)
  • Root motion global réel (pas de foot sliding)
  • Vitesse : >200 FPS
        |
        v
[CONVERTISSEUR SMPL->R15]  — Le Retargeteur
  • Sélectionne 15 os parmi les 24 joints SMPL
  • Convertit axis-angle -> quaternions WXYZ
  • Applique les conventions d'axes Roblox (X=-X_SMPL, Z=-Z_SMPL)
        |
        v
[LISSAGE + INTERPOLATION FPS]  — Le Purificateur
  • Savitzky-Golay (scipy)
  • Resampling temporel vers 30/60/120 FPS cibles
        |
        v
     .npz
  (contrat U-ALPHA -> U-GAMMA)
        |
        v
[BLENDER 4.x HEADLESS]  — La Forge
  • Applique les rotations frame par frame sur le rig R15
  • Conversion world -> local (fix v3)
  • Export FBX avec animation bakée
        |
        v
     .fbx
  (prêt pour Roblox Studio)
```

---

## Contrat .npz (Interface U-ALPHA → U-GAMMA)

```python
{
    "rotations":     np.float32,  # (N_frames, 15, 4)  quaternions WXYZ
    "root_position": np.float32,  # (N_frames, 3)      XYZ mètres
    "bone_names":    list[str],   # 15 os R15 officiels
    "fps":           int,
    "duration":      float,
    "source_fps":    int,
    "person_index":  int,
    "total_persons": int
}
```

Les 15 os R15 (ordre fixe) :
```
LowerTorso, UpperTorso, Head,
LeftUpperArm, LeftLowerArm, LeftHand,
RightUpperArm, RightLowerArm, RightHand,
LeftUpperLeg, LeftLowerLeg, LeftFoot,
RightUpperLeg, RightLowerLeg, RightFoot
```

---

## Stack Technique

| Composant | Rôle dans le pipeline |
|-----------|----------------------|
| **Gemini 2.0 Flash** (Google AI) | Analyse vidéo intelligente — pré-filtrage des segments |
| **WHAM** (CVPR 2024) | Estimation SMPL depuis vidéo monoculaire — coordonnées monde |
| **NumPy / SciPy** | Calcul quaternions, lissage Savitzky-Golay, interpolation |
| **PySceneDetect + OpenCV** | Décodage vidéo, backup détection de cuts |
| **Blender 4.x headless** | Application des rotations + export FBX |
| **Google Colab T4** | Exécution cloud gratuite |

---

## Quick Start (Google Colab)

### Etape 1 — Frégate U-ALPHA (Extraction)

1. Ouvrir `U-ALPHA/ANIMA_MECHANICUS_ALPHA.ipynb` dans Google Colab
2. Configurer ta clé API Gemini (gratuite sur [aistudio.google.com](https://aistudio.google.com))
3. Uploader une vidéo `.mp4` d'un **vrai humain**
4. Gemini analyse la vidéo et affiche un rapport de segments
5. Valider ou ajuster les segments à extraire
6. Lancer WHAM sur les segments validés
7. Télécharger les fichiers `.npz` (1 par personne)

### Etape 2 — Frégate U-GAMMA (Forge)

1. Ouvrir `U-GAMMA/ANIMA_MECHANICUS_GAMMA.ipynb` dans Google Colab
2. Uploader les fichiers `.npz` de l'étape 1
3. Lancer → Télécharger les `.fbx`
4. Importer dans Roblox Studio

---

## Contraintes

- Durée vidéo source : max 60 secondes
- Personnages : max 4 sujets simultanés
- Input requis : vidéo d'un **humain réel** (pas d'avatar 3D, pas d'animation)
- Clé Gemini API : gratuite (1M tokens/jour sur Google AI Studio)
- Clé WHAM : modèles SMPL nécessitent une inscription gratuite sur mpg.de

---

## Pourquoi ce nom ?

Dans le lore de Warhammer 40 000, les **Prêtres-Techniciens du Mechanicus** sont les gardiens de la connaissance des machines. Ils insufflent l'**Anima** (l'âme, la vie) dans les automates via des rituels de maintenance sacrés.

Ce pipeline fait de même : il capture l'**âme du mouvement humain** (Anima) et la transmute en données exploitables par la **machine** (Roblox), grâce aux rites de la forge numérique (Mechanicus).

Les deux frégates portent également des noms de l'univers 40K :
- **U-ALPHA — L'Auspex Cogitateur** : l'Auspex est le scanner universel de l'Imperium. Le Cogitateur est l'ordinateur sacré.
- **U-GAMMA — La Forge** : les Forges Mondiales où sont créées les armes et machines de guerre.

---

## Structure du Dépôt

```
ANIMA-MECHANICUS/
├── docs/
│   ├── MOTUS_VIGILUS_PRD.md       # Requirements (v1 — référence)
│   ├── MOTUS_VIGILUS_ROADMAP.md   # Feuille de route
│   └── MOTUS_VIGILUS_STATE.md     # Etat du projet
├── U-ALPHA/                       # Frégate U-ALPHA
│   ├── codebase/
│   │   └── motus_extract.py       # Script principal (Gemini + WHAM + SMPL->R15)
│   ├── inputs/                    # Vidéos .mp4 sources
│   ├── outputs/                   # Fichiers .npz extraits
│   └── ANIMA_MECHANICUS_ALPHA.ipynb
├── U-GAMMA/                       # Frégate U-GAMMA (inchangée)
│   ├── codebase/
│   │   └── motus_forge.py         # Script Blender headless
│   ├── inputs/                    # Fichiers .npz
│   ├── outputs/                   # Fichiers .fbx
│   ├── templates/
│   │   └── r15_template.blend     # Template armature R15
│   └── ANIMA_MECHANICUS_GAMMA.ipynb
├── README.md
└── requirements.txt
```

---

## Licence

Open-source — Usage libre pour projets Roblox non-commerciaux.
