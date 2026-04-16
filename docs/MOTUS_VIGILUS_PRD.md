# ANIMA-MECHANICUS — PRODUCT REQUIREMENTS DOCUMENT

## I. VISION

| Parametre | Valeur |
|-----------|--------|
| Objectif | Extraire animations depuis video humain reel → .fbx R15 compatible Roblox |
| Cout | 0€ (Colab gratuit + open-source + APIs gratuites) |
| Statut | V4 — Option D : Gemini + GVHMR + SMPL→R15 |
| Ancien nom | MOTUS-VIGILUS (abandonne — MediaPipe incompatible videos reelles) |
| Version precedente | V3 — WHAM (abandonne — dependances mmcv/detectron2 incompatibles Colab 2026) |

---

## II. ARCHITECTURE DES 2 FREGATES

| Fregate | Script | Input | Output | Stack |
|---------|--------|-------|--------|-------|
| U-ALPHA (L'Auspex Cogitateur) | motus_extract.py | .mp4 | .npz | Gemini 2.0 Flash, GVHMR, NumPy, SciPy |
| U-GAMMA (La Forge) | motus_forge.py | .npz | .fbx | Blender 4.x headless |

---

## III. PIPELINE U-ALPHA — DETAIL

### Etape 1 — Analyse Gemini

Gemini 2.0 Flash recoit la video entiere et produit un JSON structure :

```
Video .mp4
    → Gemini 2.0 Flash (API)
    → JSON : segments_valides[], segments_exclus[], camera, qualite_globale
```

Criteres de filtrage automatique des segments :
- `qualite_estimee < 0.6` → exclu
- `corps_visible = "tete_seulement"` → modele_recommande = DECA → segment skippe
- `corps_visible = "partiel"` → modele_recommande = FrankMocap_upper → GVHMR + masquage joints inferieurs
- `camera.zoom_detecte = true` → warning root motion

### Etape 2 — Extraction GVHMR

GVHMR (SIGGRAPH Asia 2024) traite chaque segment valide :

```
Segment .mp4 decoupe
    → GVHMR inference (tools/demo/demo.py)
    → poses : (N_frames, 24, 3)   axis-angle SMPL
    → transl : (N_frames, 3)      translation monde (gravity-view coordinates)
```

Caracteristiques cles :
- Coordonnees MONDE avec repere gravite-vue — root motion reel superieur a WHAM
- Detection personnes : YOLOv8 (ultralytics)
- Pose 2D : ViTPose-H
- HMR backbone : HMR2.0a
- Multi-personnes : 1 passe GVHMR par person_id

Cas corps partiel (modele_recommande = FrankMocap_upper) :
- GVHMR tourne normalement
- Post-traitement : joints inferieurs non visibles mis a quaternion identite [1,0,0,0]
- Joints masques : LeftUpperLeg, LeftLowerLeg, LeftFoot, RightUpperLeg, RightLowerLeg, RightFoot

### Etape 3 — Retargeting SMPL→R15

Mapping fixe des 24 joints SMPL vers les 15 os R15 :

| SMPL joint | R15 os |
|-----------|--------|
| Pelvis (0) | LowerTorso |
| Spine3 (9) | UpperTorso |
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

Note : le code utilise Spine3 (index 9) → UpperTorso. Le PRD precedent indiquait Spine2 (6), le code fait foi.

Conversion : axis-angle → rotation matrix → quaternion WXYZ
Application convention axes Roblox : X_roblox = -X_SMPL, Z_roblox = -Z_SMPL

### Etape 4 — Lissage + Interpolation FPS

- Lissage Savitzky-Golay (scipy) — 3 niveaux : faible / moyen / fort
- Renforcement sur extremites (mains, pieds, avant-bras)
- Resampling temporel vers FPS cible (30 / 60 / 120)
- Interpolation lineaire des gaps d'occlusion (max 10 frames)

---

## IV. CONTRAT .npz (Interface U-ALPHA → U-GAMMA)

```
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
LowerTorso, UpperTorso, Head,
LeftUpperArm, LeftLowerArm, LeftHand,
RightUpperArm, RightLowerArm, RightHand,
LeftUpperLeg, LeftLowerLeg, LeftFoot,
RightUpperLeg, RightLowerLeg, RightFoot

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
          "modele_recommande": "GVHMR|FrankMocap_upper|DECA|skip",
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

Note : valeur `GVHMR` = pipeline complet. `FrankMocap_upper` = GVHMR + masquage joints inferieurs. `DECA` = segment skippe (tete seulement, non supporte). `skip` = segment exclu.

---

## VI. STACK TECHNIQUE

| Composant | Version | Usage |
|-----------|---------|-------|
| Python | 3.10 | Orchestration |
| Gemini | 2.0 Flash | Analyse video intelligente |
| GVHMR | SIGGRAPH Asia 2024 | Estimation pose SMPL monde (remplace WHAM) |
| YOLOv8 | ultralytics 8.2.42 | Detection personnes (interne GVHMR) |
| ViTPose-H | multi-coco | Pose 2D (interne GVHMR) |
| HMR2.0a | epoch=10-step=25000 | Regression mesh SMPL (interne GVHMR) |
| NumPy | 1.23.5 (GVHMR pin) | Calcul quaternions, axes |
| SciPy | Latest | Smoothing Savitzky-Golay + interpolation |
| OpenCV | Latest | Decodage video, decoupe segments |
| Blender | 4.0+ | Export FBX headless (U-GAMMA) |
| Google Colab | T4 GPU | Execution cloud |

---

## VII. CONTRAINTES

- Input : video d'un humain reel uniquement (pas d'avatar 3D, pas d'animation)
- Duree max : 60 secondes de video source
- Personnages : max 4 sujets simultanees
- Etancheite : U-ALPHA ignore Blender. U-GAMMA ignore Gemini/GVHMR.
- Cle Gemini : gratuite sur aistudio.google.com (1M tokens/jour)
- Modeles SMPL/SMPLX : disponibles sur HuggingFace (camenduru) sans inscription

---

## VIII. METRIQUES DE SUCCES

| Metrique | Cible |
|----------|-------|
| Temps traitement (1 min video) | < 15 min (Colab T4) |
| Qualite animation | Pas de joint popping, root motion coherent |
| Filtrage Gemini | 0 segment degrade envoye a GVHMR |
| Code source U-ALPHA | < 700 lignes |
| Cout recurrent | 0€ |
| Installation Colab | Reussit du premier coup sans erreur |
