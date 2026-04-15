# ANIMA-MECHANICUS

> "L'Âme du Mouvement, Forgée par la Machine" — Pipeline de Motion Capture Vidéo → FBX Roblox R15

Inspiré du lore Warhammer 40 000 — les Prêtres-Techniciens du Dieu-Machine qui insufflent la vie dans les automates.

---

## Vision

**ANIMA-MECHANICUS** extrait les mouvements d'une vidéo de **vrai humain** et les convertit en fichier `.fbx` compatible avec les avatars Roblox R15.

Pipeline 100% gratuit, exécuté sur Google Colab.

Anciennement connu sous le nom de **MOTUS-VIGILUS** (MediaPipe). Cette version V4 remplace le moteur d'extraction par une chaîne académique de pointe : **Gemini 2.0 Flash** (analyse intelligente) + **GVHMR** (estimation de pose SMPL, SIGGRAPH Asia 2024).

---

## Architecture — Les 2 Frégates

| Frégate | Script | Rôle | Input | Output |
|---------|--------|------|-------|--------|
| **U-ALPHA** (L'Auspex Cogitateur) | `U-ALPHA/codebase/motus_extract.py` | Analyse Gemini + Extraction GVHMR + Transmutation SMPL→R15 | `.mp4` | `.npz` |
| **U-GAMMA** (La Forge) | `U-GAMMA/codebase/motus_forge.py` | Manifestation FBX via Blender headless | `.npz` | `.fbx` |

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
  • Retourne un rapport JSON : segments, qualité, routing modèle
        |
        v  (segments validés uniquement)
[GVHMR]  — Le Détecteur de Chair
  • Moteur de pose SMPL de niveau académique (SIGGRAPH Asia 2024)
  • Reconstruit 24 joints en coordonnées MONDE avec repère gravité-vue
  • Root motion global réel (pas de foot sliding)
  • Zéro mmcv — Zéro detectron2 — PyTorch CUDA 12.1
  • Détection : YOLOv8 | Pose 2D : ViTPose-H | Backbone : HMR2.0a
        |
        v
[CONVERTISSEUR SMPL→R15]  — Le Retargeteur
  • Sélectionne 15 os parmi les 24 joints SMPL
  • Convertit axis-angle → quaternions WXYZ
  • Applique les conventions d'axes Roblox (X=-X_SMPL, Z=-Z_SMPL)
  • Corps partiel (FrankMocap_upper) : joints inférieurs → identité [1,0,0,0]
        |
        v
[LISSAGE + INTERPOLATION FPS]  — Le Purificateur
  • Savitzky-Golay (scipy)
  • Resampling temporel vers 30/60/120 FPS cibles
        |
        v
     .npz
  (contrat U-ALPHA → U-GAMMA)
        |
        v
[BLENDER 4.x HEADLESS]  — La Forge
  • Applique les rotations frame par frame sur le rig R15
  • Conversion world → local (fix v3)
  • Export FBX avec animation bakée
        |
        v
     .fbx
  (prêt pour Roblox Studio)
```

---

## Contrat JSON Gemini (Sortie de l'Analyse Vidéo)

Gemini analyse la vidéo entière et produit ce JSON avant tout traitement GVHMR.
Seuls les segments avec `qualite_estimee >= 0.6` sont envoyés à GVHMR.

```json
{
  "video_duration_seconds": 45.2,
  "source_fps": 30,
  "total_persons": 2,

  "persons": [
    {
      "person_id": 0,
      "segments_valides": [
        {
          "start_s": 2.5,
          "end_s": 38.0,
          "corps_visible": "complet",
          "orientation": "face",
          "distance_camera": "proche",
          "type_mouvement": "danse",
          "qualite_estimee": 0.9,
          "problemes": [],
          "extraction_possible": "corps_complet",
          "modele_recommande": "GVHMR"
        },
        {
          "start_s": 41.0,
          "end_s": 45.2,
          "corps_visible": "tronc",
          "orientation": "profil",
          "distance_camera": "moyen",
          "type_mouvement": "marche",
          "qualite_estimee": 0.7,
          "problemes": ["jambes_coupees"],
          "extraction_possible": "haut_du_corps",
          "modele_recommande": "FrankMocap_upper"
        }
      ],
      "segments_exclus": [
        {
          "start_s": 0.0,
          "end_s": 2.5,
          "raison": "personne_absente"
        },
        {
          "start_s": 38.0,
          "end_s": 41.0,
          "raison": "flou_mouvement"
        }
      ]
    }
  ],

  "camera": {
    "mouvement": "stable",
    "zoom_detecte": false
  },

  "qualite_globale": "bonne",
  "recommandation": "Extraire personne 0 sur segment 2.5-38.0s. Corps complet visible, caméra stable."
}
```

### Routing automatique par segment

| `modele_recommande` | Condition | Traitement GVHMR |
|---------------------|-----------|-----------------|
| `GVHMR` (ou `WHAM` legacy) | `corps_visible=complet` + `qualite >= 0.6` | Extraction complète |
| `FrankMocap_upper` | `corps_visible=tronc` + `qualite >= 0.6` | GVHMR + masquage joints inférieurs → identité |
| `DECA` | `corps_visible=tete_seulement` + `qualite >= 0.5` | Ignoré (non implémenté) |
| `skip` | `qualite < seuil` | Exclu |

### Valeurs possibles par champ

| Champ | Valeurs | Impact sur le pipeline |
|-------|---------|------------------------|
| `corps_visible` | `complet` / `tronc` / `tete_seulement` | `tronc` = masquage jambes ; `tete_seulement` = ignoré |
| `orientation` | `face` / `profil` / `dos` / `mixte` | `dos` = warning précision bras réduite |
| `distance_camera` | `proche` / `moyen` / `lointain` | `lointain` = warning qualité GVHMR dégradée |
| `type_mouvement` | `marche` / `danse` / `combat` / `sport` / `statique` | Info méta uniquement |
| `qualite_estimee` | `0.0` → `1.0` | Seuil de filtrage automatique : `< 0.6` = exclu |
| `extraction_possible` | `corps_complet` / `haut_du_corps` / `tete_cou` / `aucune` | Détermine le mode d'extraction |
| `problemes` | `jambes_coupees` / `flou_mouvement` / `occlusion_partielle` / `contre_jour` | Affiché comme warning |
| `camera.mouvement` | `stable` / `panoramique` / `suivi` / `agitee` | `agitee` = root motion peu fiable |
| `camera.zoom_detecte` | `true` / `false` | `true` = root motion faussé, warning |

### Ce que Gemini N'extrait PAS (volontairement)

| Donnée | Raison de l'exclusion |
|--------|-----------------------|
| Position exacte caméra / focale | GVHMR est monoculaire, non nécessaire |
| Description du décor | Aucun impact sur la pose |
| Description des vêtements | Aucun impact |
| Landmarks 2D manuels | GVHMR les calcule mieux lui-même (ViTPose-H) |
| Audio / parole | Hors scope |

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
| **Gemini 2.0 Flash** (Google AI) | Analyse vidéo intelligente — pré-filtrage et routing des segments |
| **GVHMR** (SIGGRAPH Asia 2024) | Estimation SMPL depuis vidéo monoculaire — coordonnées monde |
| **YOLOv8** (ultralytics) | Détection personnes (interne GVHMR) |
| **ViTPose-H** | Estimation pose 2D (interne GVHMR) |
| **HMR2.0a** | Régression mesh SMPL (interne GVHMR) |
| **NumPy / SciPy** | Calcul quaternions, lissage Savitzky-Golay, interpolation |
| **OpenCV** | Décodage vidéo, découpe segments |
| **Blender 4.x headless** | Application des rotations + export FBX |
| **Google Colab T4** | Exécution cloud gratuite (CUDA 12.1) |

---

## Quick Start (Google Colab)

### Etape 1 — Frégate U-ALPHA (Extraction)

1. Ouvrir `U-ALPHA/ANIMA_MECHANICUS_ALPHA.ipynb` dans Google Colab
2. **Cellule 1** — Installer les dépendances (GVHMR, Torch CUDA 12.1, google-genai)
3. **Cellule 1b** — Télécharger SMPL_NEUTRAL.pkl depuis HuggingFace *(si manquant au Pre-Flight)*
4. **Cellule 2** — Configurer ta clé API Gemini (gratuite sur [aistudio.google.com](https://aistudio.google.com))
5. Uploader une vidéo `.mp4` d'un **vrai humain**
6. Gemini analyse la vidéo et affiche un rapport de segments
7. Valider ou ajuster les segments à extraire
8. Lancer GVHMR sur les segments validés
9. Télécharger les fichiers `.npz` (1 par personne)

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
- Modèles SMPL : disponibles sur HuggingFace (camenduru) sans inscription

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
│   ├── MOTUS_VIGILUS_PRD.md       # Requirements (V4 — GVHMR)
│   ├── MOTUS_VIGILUS_ROADMAP.md   # Feuille de route
│   └── MOTUS_VIGILUS_STATE.md     # Etat du projet
├── U-ALPHA/                       # Frégate U-ALPHA
│   ├── codebase/
│   │   └── motus_extract.py       # Script principal (Gemini + GVHMR + SMPL→R15)
│   ├── inputs/                    # Vidéos .mp4 sources
│   ├── outputs/                   # Fichiers .npz extraits
│   └── ANIMA_MECHANICUS_ALPHA.ipynb
├── U-GAMMA/                       # Frégate U-GAMMA
│   ├── codebase/
│   │   └── motus_forge.py         # Script Blender headless (rig R15 généré programmatiquement)
│   ├── inputs/                    # Fichiers .npz
│   ├── outputs/                   # Fichiers .fbx
│   └── ANIMA_MECHANICUS_GAMMA.ipynb
├── README.md
└── requirements.txt
```

---

## Historique des Versions

| Version | Stack | Statut |
|---------|-------|--------|
| V1 | 3 Frégates + BVH | Abandonné — sur-ingénierie |
| V2 | MediaPipe | Échoué — incompatible vidéos réelles |
| V3 | Gemini + WHAM | Bloqué — mmcv/detectron2 incompatibles Colab 2026 |
| **V4** | **Gemini + GVHMR** | **En cours — Phase 6** |

---

## Licence

Open-source — Usage libre pour projets Roblox non-commerciaux.
