# ANIMA-MECHANICUS — PHYLACTERE DE RESURRECTION

| Champ | Valeur |
|-------|--------|
| STATUS | Phase 2 — En cours (U-ALPHA v2 à coder) |
| DATE | Avril 2026 |
| ARCHITECTURE | 2 Frégates (U-ALPHA + U-GAMMA) |
| VERSION | V3 — Option C : Gemini 2.0 Flash + WHAM + SMPL→R15 |
| CONTRAT .npz | Inchangé : rotations (N,15,4) + root_position (N,3) + metadata |
| CONTRAT JSON GEMINI | Défini et documenté dans README |

---

## [LAST_WORK]

- Projet renommé ANIMA-MECHANICUS (WH40K lore)
- Depot GitHub créé : kioka8877-ux/ANIMA-MECHANICUS
- Architecture pivot : MediaPipe abandonné → Gemini 2.0 Flash + WHAM + SMPL→R15
- README v2 rédigé avec pipeline complet Option C
- Contrat JSON Gemini défini et documenté (segments, qualité, caméra, warnings)
- U-GAMMA (motus_forge.py v3) : fonctionnel, inchangé
- motus_extract.py actuel : version MediaPipe obsolète (à remplacer)

---

## [NEXT_TASK]

Réécrire `U-ALPHA/codebase/motus_extract.py` avec le pipeline Option C :

1. Module Gemini — analyse vidéo → JSON segments
2. Module WHAM — estimation SMPL sur segments validés
3. Module SMPL→R15 — retargeting 24 joints → 15 os R15
4. Module lissage + interpolation FPS (conserver de l'ancienne version)
5. Export .npz (conserver le contrat existant)

---

## [BLOCKERS]

- Modèles SMPL (body_models/) nécessitent inscription gratuite sur mpg.de
  → Documenter la procédure dans le notebook Colab
- WHAM nécessite ffmpeg + torch + detectron2 sur Colab T4
  → Prévoir cellule d'installation dans le notebook

---

## [SOLUTIONS]

| Problème | Solution |
|----------|---------|
| Camera agitée → root motion faussé | Warning Gemini `camera.mouvement = agitee` + option désactivation root_position |
| Corps partiel → hallucination WHAM | Filtre Gemini `qualite_estimee < 0.6` → segment exclu automatiquement |
| Occlusions courtes (<10 frames) | Interpolation linéaire scipy interp1d (conservé) |
| SMPL 24 joints → R15 15 os | Mapping fixe défini dans le convertisseur |
| Blender headless lent | Template R15 pré-chargé en .blend (inchangé U-GAMMA) |
| Zoom vidéo → root motion faussé | Warning Gemini `camera.zoom_detecte = true` |

---

## [HISTORIQUE DES VERSIONS]

| Version | Architecture | Statut |
|---------|-------------|--------|
| V1 | 3 Frégates + BVH | Abandonné |
| V2 | 2 Frégates + MediaPipe | Echoué (vidéos Roblox 3D incompatibles) |
| V3 | 2 Frégates + Gemini + WHAM | En cours |
