# ANIMA-MECHANICUS — PRODUCT REQUIREMENTS DOCUMENT

## I. VISION

| Paramètre | Valeur |
|-----------|--------|
| Objectif | Extraire animations depuis vidéo humain réel → .fbx R15 compatible Roblox |
| Coût | 0€ (Colab gratuit + open-source + APIs gratuites) |
| Statut | V3 — Option C : Gemini + WHAM + SMPL→R15 |
| Ancien nom | MOTUS-VIGILUS (abandonné — MediaPipe incompatible vidéos réelles) |

---

## II. ARCHITECTURE DES 2 FREGATES

| Frégate | Script | Input | Output | Stack |
|---------|--------|-------|--------|-------|
| U-ALPHA (L'Auspex Cogitateur) | motus_extract.py | .mp4 | .npz | Gemini 2.0 Flash, WHAM, NumPy, SciPy |
| U-GAMMA (La Forge) | motus_forge.py | .npz + .blend | .fbx | Blender 4.x headless |

---

## III. PIPELINE U-ALPHA — DETAIL

### Etape 1 — Analyse Gemini

Gemini 2.0 Flash reçoit la vidéo entière et produit un JSON structuré :

```
Vidéo .mp4
    → Gemini 2.0 Flash (API)
    → JSON : segments_valides[], segments_exclus[], camera, qualite_globale
```

Critères de filtrage automatique des segments :
- `qualite_estimee < 0.6` → exclu
- `corps_visible != "complet"` → warning ou exclu selon niveau
- `camera.zoom_detecte = true` → warning root motion

### Etape 2 — Extraction WHAM

WHAM (CVPR 2024) traite chaque segment validé :

```
Segment .mp4 découpé
    → WHAM inference
    → poses : (N_frames, 24, 3)   axis-angle SMPL
    → transl : (N_frames, 3)      translation monde
```

Caractéristiques clés :
- Coordonnées MONDE (pas caméra) — root motion réel sans foot sliding
- Vitesse >200 FPS sur T4
- Multi-personnes : 1 passe WHAM par person_id

### Etape 3 — Retargeting SMPL→R15

Mapping fixe des 24 joints SMPL vers les 15 os R15 :

| SMPL joint | R15 os |
|-----------|--------|
| Pelvis (0) | LowerTorso |
| Spine2 (6) | UpperTorso |
| Head (15) | Head |
| L_Shoulder (16) | LeftUpperArm |
| L_Elbow (18) | LeftLowerArm |
| L_Wrist (20) | LeftHand |
| R_Shoulder (17) | RightUpperArm |
| R_Elbow (19) | RightLowerArm |
| R_Wrist (21) | RightHand |
| L_Hip (1) | LeftUpperLeg |
| L_Knee (4) | LeftLowerLeg |
| L_Ankle (7) | LeftFoot |
| R_Hip (2) | RightUpperLeg |
| R_Knee (5) | RightLowerLeg |
| R_Ankle (8) | RightFoot |

Conversion : axis-angle → rotation matrix → quaternion WXYZ
Application convention axes Roblox : X_roblox = -X_SMPL, Z_roblox = -Z_SMPL

### Etape 4 — Lissage + Interpolation FPS

- Lissage Savitzky-Golay (scipy) — 3 niveaux : faible / moyen / fort
- Renforcement sur extrémités (mains, pieds, avant-bras)
- Resampling temporel vers FPS cible (30 / 60 / 120)
- Interpolation linéaire des gaps d'occlusion (max 10 frames)

---

## IV. CONTRAT .npz (Interface U-ALPHA → U-GAMMA)

```python
{
    "rotations":     np.float32,  # (N_frames, 15, 4)  quaternions WXYZ
    "root_position": np.float32,  # (N_frames, 3)      XYZ metres
    "bone_names":    list[str],   # 15 os R15 ordre fixe
    "fps":           int,
    "duration":      float,
    "source_fps":    int,
    "person_index":  int,
    "total_persons": int
}
```

Ordre des 15 os (immuable) :
```
LowerTorso, UpperTorso, Head,
LeftUpperArm, LeftLowerArm, LeftHand,
RightUpperArm, RightLowerArm, RightHand,
LeftUpperLeg, LeftLowerLeg, LeftFoot,
RightUpperLeg, RightLowerLeg, RightFoot
```

---

## V. CONTRAT JSON GEMINI

```json
{
  "video_duration_seconds": float,
  "source_fps": int,
  "total_persons": int,
  "persons": [
    {
      "person_id": int,
      "segments_valides": [
        {
          "start_s": float,
          "end_s": float,
          "corps_visible": "complet|partiel|tete_seulement",
          "orientation": "face|profil|dos|mixte",
          "distance_camera": "proche|moyen|lointain",
          "type_mouvement": "marche|danse|combat|sport|statique",
          "qualite_estimee": float,
          "problemes": []
        }
      ],
      "segments_exclus": [
        {
          "start_s": float,
          "end_s": float,
          "raison": "personne_absente|flou_mouvement|occlusion_totale|contre_jour"
        }
      ]
    }
  ],
  "camera": {
    "mouvement": "stable|panoramique|suivi|agitee",
    "zoom_detecte": bool
  },
  "qualite_globale": "excellente|bonne|moyenne|mauvaise",
  "recommandation": "string"
}
```

---

## VI. STACK TECHNIQUE

| Composant | Version | Usage |
|-----------|---------|-------|
| Python | 3.10+ | Orchestration |
| Gemini | 2.0 Flash | Analyse vidéo intelligente |
| WHAM | CVPR 2024 | Estimation pose SMPL monde |
| NumPy | Latest | Calcul quaternions, axes |
| SciPy | Latest | Smoothing Savitzky-Golay + interpolation |
| OpenCV | Latest | Décodage vidéo, découpe segments |
| Blender | 4.0+ | Export FBX headless (U-GAMMA) |
| Google Colab | T4 GPU | Exécution cloud |

---

## VII. CONTRAINTES

- Input : vidéo d'un humain réel uniquement (pas d'avatar 3D, pas d'animation)
- Durée max : 60 secondes de vidéo source
- Personnages : max 4 sujets simultanés
- Etanchéité : U-ALPHA ignore Blender. U-GAMMA ignore Gemini/WHAM.
- Clé Gemini : gratuite sur aistudio.google.com (1M tokens/jour)
- Modèles SMPL : inscription gratuite requise sur mpg.de

---

## VIII. METRIQUES DE SUCCES

| Métrique | Cible |
|----------|-------|
| Temps traitement (1 min vidéo) | < 15 min (Colab T4) |
| Qualité animation | Pas de joint popping, root motion cohérent |
| Filtrage Gemini | 0 segment dégradé envoyé à WHAM |
| Code source U-ALPHA | < 600 lignes |
| Coût récurrent | 0€ |
