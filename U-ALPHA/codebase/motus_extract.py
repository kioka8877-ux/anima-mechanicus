#!/usr/bin/env python3
"""ANIMA-MECHANICUS — Frégate U-ALPHA : L'Auspex Cogitateur — v3
Pipeline : Gemini (analyse + cascade anti-429) → WHAM (SMPL) → SMPL→R15 → .npz

Usage:
    python motus_extract.py video.mp4 --gemini-keys CLE1,CLE2 [options]

Options:
    --gemini-key  CLE                Cle unique (alias de --gemini-keys)
    --gemini-keys CLE1,CLE2,...      Plusieurs cles separees par virgule (rotation auto sur 429)
    --fps {30,60,120}                FPS cible (défaut: 30)
    --smooth {faible,moyen,brutal}   Niveau lissage (défaut: moyen)
    --no-root-motion                 Désactive la translation globale
    --quality-threshold F            Score Gemini minimum (défaut: 0.6)
    --output DIR                     Dossier de sortie (défaut: outputs/)
    --wham-dir DIR                   Chemin vers le dépôt WHAM (défaut: ~/WHAM)
    --cache-dir DIR                  Cache résultats Gemini (évite re-analyse, défaut: désactivé)
    --start-model N                  Démarre la cascade au modèle N (0=pro, 1=2.5flash, 2=2.0flash, 3=1.5flash)
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import pickle
import time
import hashlib
from pathlib import Path

import numpy as np
import cv2
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

BONE_NAMES = [
    "LowerTorso", "UpperTorso", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]

# Mapping SMPL joint index → nom os R15
# SMPL 24 joints : 0=Pelvis, 1=L_Hip, 2=R_Hip, 3=Spine1, 4=L_Knee, 5=R_Knee,
#                  6=Spine2, 7=L_Ankle, 8=R_Ankle, 9=Spine3, 10=L_Toe, 11=R_Toe,
#                  12=Neck, 13=L_Collar, 14=R_Collar, 15=Head,
#                  16=L_Shoulder, 17=R_Shoulder, 18=L_Elbow, 19=R_Elbow,
#                  20=L_Wrist, 21=R_Wrist, 22=L_Hand, 23=R_Hand
SMPL_TO_R15 = {
    0:  "LowerTorso",
    9:  "UpperTorso",     # Spine3 = sommet du torse
    15: "Head",
    16: "LeftUpperArm",
    18: "LeftLowerArm",
    20: "LeftHand",
    17: "RightUpperArm",
    19: "RightLowerArm",
    21: "RightHand",
    1:  "LeftUpperLeg",
    4:  "LeftLowerLeg",
    7:  "LeftFoot",
    2:  "RightUpperLeg",
    5:  "RightLowerLeg",
    8:  "RightFoot",
}

R15_INDEX = {name: i for i, name in enumerate(BONE_NAMES)}

SMOOTH_PRESETS = {"faible": (5, 2), "moyen": (7, 3), "brutal": (15, 3)}

EXTREMITY_BONE_NAMES = {
    "LeftLowerArm", "LeftHand",
    "RightLowerArm", "RightHand",
    "LeftLowerLeg", "LeftFoot",
    "RightLowerLeg", "RightFoot",
}
EXTREMITY_INDICES = [i for i, name in enumerate(BONE_NAMES) if name in EXTREMITY_BONE_NAMES]

GEMINI_PROMPT = """Tu es un analyseur de vidéo expert en motion capture humaine.
Analyse la vidéo et retourne UNIQUEMENT un bloc JSON valide, sans markdown, sans explication.
Commence directement par { et termine par }.

FORMAT EXACT ATTENDU :
{
  "video_duration_seconds": <float>,
  "source_fps": <int>,
  "total_persons": <int>,
  "persons": [
    {
      "person_id": <int>,
      "segments_valides": [
        {
          "start_s": <float>,
          "end_s": <float>,
          "corps_visible": "<EXACTEMENT : complet | tronc | tete_seulement>",
          "orientation": "<EXACTEMENT : face | profil | dos | mixte>",
          "distance_camera": "<EXACTEMENT : proche | moyen | lointain>",
          "type_mouvement": "<EXACTEMENT : marche | danse | combat | sport | statique | discussion>",
          "qualite_estimee": <float 0.0-1.0>,
          "problemes": [],
          "extraction_possible": "<EXACTEMENT : corps_complet | haut_du_corps | tete_cou | aucune>",
          "modele_recommande": "<EXACTEMENT : WHAM | FrankMocap_upper | DECA | skip>"
        }
      ],
      "segments_exclus": [
        {
          "start_s": <float>,
          "end_s": <float>,
          "raison": "<EXACTEMENT : personne_absente | flou_mouvement | occlusion_totale | contre_jour | trop_court>"
        }
      ]
    }
  ],
  "camera": {
    "mouvement": "<EXACTEMENT : stable | panoramique | suivi | agitee>",
    "zoom_detecte": <bool>
  },
  "qualite_globale": "<EXACTEMENT : excellente | bonne | moyenne | mauvaise>",
  "recommandation": "<string>"
}

RÈGLES DE VISIBILITÉ :
- corps_visible "complet"       = tête + torse + bras + jambes tous visibles
- corps_visible "tronc"         = tête + torse + bras visibles, jambes absentes ou coupées
- corps_visible "tete_seulement" = seule la tête ou le buste sans bras (plan serré visage)

RÈGLES DE ROUTING (extraction_possible + modele_recommande) :
- complet   + qualite >= 0.6 → extraction_possible "corps_complet"  + modele_recommande "WHAM"
- tronc     + qualite >= 0.6 → extraction_possible "haut_du_corps"  + modele_recommande "FrankMocap_upper"
- tete_seulement + qualite >= 0.5 → extraction_possible "tete_cou"  + modele_recommande "DECA"
- qualite sous seuil          → extraction_possible "aucune"         + modele_recommande "skip"

RÈGLES DE SEGMENTATION :
- Durée minimale d'un segment_valide : 1.5 secondes
- Segment < 1.5s → fusionner avec adjacent de même type, ou segments_exclus avec raison "trop_court"
- Découper en segments distincts si la visibilité change (ex : plan large → gros plan)
- Les timestamps doivent couvrir toute la durée (pas de trous non justifiés)
- Un gros plan visage est NORMAL dans une vidéo de discussion → segment tete_seulement valide, pas exclu

RÈGLES QUALITÉ :
- qualite_estimee 1.0 = corps complet, face caméra, lumière parfaite, sans flou
- qualite_estimee 0.6 = seuil minimum pour complet et tronc
- qualite_estimee 0.5 = seuil minimum pour tete_seulement
- zoom_detecte = true uniquement si la caméra zoome/dé-zoome visiblement
- Créer un person_id distinct par personne présente dans la vidéo
"""


# Schema JSON attendu de Gemini (technique response_schema = JSON garanti valide,
# sans markdown, sans hallucination de structure — inspire de EXO_00_CORTEX)
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "video_duration_seconds": {"type": "NUMBER"},
        "source_fps":             {"type": "INTEGER"},
        "total_persons":          {"type": "INTEGER"},
        "persons": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "person_id": {"type": "INTEGER"},
                    "segments_valides": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_s":             {"type": "NUMBER"},
                                "end_s":               {"type": "NUMBER"},
                                "corps_visible":       {"type": "STRING",
                                                       "enum": ["complet", "tronc", "tete_seulement"]},
                                "orientation":         {"type": "STRING",
                                                       "enum": ["face", "profil", "dos", "mixte"]},
                                "distance_camera":     {"type": "STRING",
                                                       "enum": ["proche", "moyen", "lointain"]},
                                "type_mouvement":      {"type": "STRING",
                                                       "enum": ["marche", "danse", "combat", "sport",
                                                                "statique", "discussion"]},
                                "qualite_estimee":     {"type": "NUMBER"},
                                "problemes":           {"type": "ARRAY",
                                                       "items": {"type": "STRING"}},
                                "extraction_possible": {"type": "STRING",
                                                       "enum": ["corps_complet", "haut_du_corps",
                                                                "tete_cou", "aucune"]},
                                "modele_recommande":   {"type": "STRING",
                                                       "enum": ["WHAM", "FrankMocap_upper", "DECA", "skip"]},
                            },
                            "required": ["start_s", "end_s", "corps_visible", "orientation",
                                         "distance_camera", "type_mouvement", "qualite_estimee",
                                         "problemes", "extraction_possible", "modele_recommande"],
                        }
                    },
                    "segments_exclus": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_s": {"type": "NUMBER"},
                                "end_s":   {"type": "NUMBER"},
                                "raison":  {"type": "STRING",
                                           "enum": ["personne_absente", "flou_mouvement",
                                                    "occlusion_totale", "contre_jour", "trop_court"]},
                            },
                            "required": ["start_s", "end_s", "raison"],
                        }
                    },
                },
                "required": ["person_id", "segments_valides", "segments_exclus"],
            }
        },
        "camera": {
            "type": "OBJECT",
            "properties": {
                "mouvement":     {"type": "STRING",
                                 "enum": ["stable", "panoramique", "suivi", "agitee"]},
                "zoom_detecte":  {"type": "BOOLEAN"},
            },
            "required": ["mouvement", "zoom_detecte"],
        },
        "qualite_globale":  {"type": "STRING",
                            "enum": ["excellente", "bonne", "moyenne", "mauvaise"]},
        "recommandation":   {"type": "STRING"},
    },
    "required": ["video_duration_seconds", "source_fps", "total_persons",
                 "persons", "camera", "qualite_globale", "recommandation"],
}


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1 — ANALYSE GEMINI
# ──────────────────────────────────────────────────────────────────────────────

# Familles de modeles par ordre de preference — du plus puissant au plus permissif.
# Modeles actifs en 2026 (gemini-2.0-flash et gemini-1.5-flash sont DEPRECIES/retires) :
#
#   Famille              | Free tier          | Tier 1 (billing lie, toujours gratuit)
#   ─────────────────────┼────────────────────┼────────────────────────────────────────
#   gemini-2.5-pro       | 5 RPM  / 25 RPD   | 1 000 RPM / ~10 000 RPD
#   gemini-2.5-flash     | 10 RPM / 500 RPD  | 2 000 RPM / 10 000 RPD
#   gemini-2.5-flash-lite| 15 RPM / 1 000 RPD| 4 000 RPM / 14 000 RPD  ← RECOMMANDE
#   gemini-1.5-pro       | 2 RPM  / 50 RPD   | fallback legacy
#
# Pour passer au Tier 1 (gratuit, 200x plus de quota) :
#   1. Creer une cle sur https://aistudio.google.com/apikey
#   2. Lier un compte de facturation : https://console.cloud.google.com/billing
#   3. La carte (meme prepayee Revolut/N26) sert uniquement de verification — 0€ preleve
#
# NOTE : les noms exacts sont resolus dynamiquement depuis ListModels
#        pour s'adapter aux modeles reellement disponibles sur le compte.
_GEMINI_FAMILIES = [
    "gemini-2.5-flash-lite-preview",  # DEPART 2.5 lite (alias stable)
    "gemini-2.5-flash",               # Fallback 2.5 flash
    "gemini-2.0-flash-lite",          # Confirme disponible (alias gemini-2.0-flash-lite-001)
    "gemini-2.0-flash",               # 2.0 flash stable
    "gemini-1.5-pro",                 # Fallback legacy
]

# Sous-chaines a exclure lors du matching : modeles non-video ou sous-optimaux
# NOTE : "-lite" est intentionnellement ABSENT — gemini-2.5-flash-lite est valide
_AVOID_IN_MODEL = (
    "-audio", "-native-audio", "-tts", "-embedding",
    "-aqa", "-nano", "-it",
)

# Suffixes de modeles non-vision a exclure dans ListModels
_EXCLUDED_SUFFIXES = ("-tts", "-audio", "-embedding", "-aqa", "-it")

# Attente minimale RPM : 65s pour vider la fenetre d'une minute
_RPM_WAIT_BASE = 65

# Videos sous ce seuil → mode inline : 1 seule requete (pas d'upload ni de polling)
_MAX_INLINE_MB = 15


def _list_vision_models(client) -> list:
    """Retourne la liste des modeles Gemini supportant la vision video."""
    try:
        all_models = [m.name.replace("models/", "") for m in client.models.list()]
        return [
            m for m in all_models
            if not any(m.endswith(s) for s in _EXCLUDED_SUFFIXES)
            and not any(bad in m for bad in ("-native-audio",))
        ]
    except Exception:
        return []


def _resolve_model(candidate: str, available: list) -> str:
    """Trouve le meilleur modele video correspondant au candidat.

    Priorite :
      1. Correspondance exacte
      2. Correspondance prefixe en excluant les variants non-video
         (audio, lite, nano...) — retourne le NOM LE PLUS COURT (plus stable)
      3. Correspondance prefixe sans restriction — retourne le plus court
      4. Nom exact tel quel (peut 404, mais derniere chance)
    """
    # 1. Exact
    if candidate in available:
        return candidate

    prefix = candidate + "-"

    # 2. Prefixe strict : exclure les variants non-video
    good = [
        m for m in available
        if m.startswith(prefix) and not any(bad in m for bad in _AVOID_IN_MODEL)
    ]
    if good:
        return sorted(good, key=len)[0]  # plus court = plus canonical/stable

    # 3. Prefixe sans restriction (fallback)
    any_match = [m for m in available if m.startswith(prefix)]
    if any_match:
        return sorted(any_match, key=len)[0]

    # 4. Aucun match — retourner le nom exact (tentera l'appel API directement)
    return candidate


def _build_cascade_from_available(available: list, start: int = 0) -> list:
    """Construit la cascade de modeles video depuis la liste reelle des modeles disponibles.

    Parcourt _GEMINI_FAMILIES dans l'ordre de preference et resout chaque famille
    vers le meilleur modele reel disponible.  Ne retient chaque modele resolu qu'une
    seule fois (dedoublonnage).

    Args:
        available : liste brute des modeles (depuis ListModels)
        start     : index de depart dans _GEMINI_FAMILIES (pour --start-model)
    """
    cascade = []
    seen = set()
    families = _GEMINI_FAMILIES[start:]

    for family in families:
        resolved = _resolve_model(family, available)
        if resolved and resolved not in seen:
            cascade.append(resolved)
            seen.add(resolved)

    if not cascade:
        # Dernier recours : utiliser les 4 premiers modeles disponibles
        cascade = available[:4]

    return cascade


def _classify_429(err_str: str) -> str:
    """Classifie le type d'erreur 429 pour adapter la strategie de retry.

    Returns :
        'rpd' — quota journalier epuise → skip immediat vers le modele suivant
        'rpm' — limite par minute     → attendre 65s+ avant de retenter

    Logique :
        RPD : "per_day", "daily", "exhausted", "quota_exceeded", "resource_exhausted"
              sans precision de fenetre temporelle (= quota journalier ou total epuise)
        RPM : "per_minute", "rpm", mentions explicites de fenetre 1 min
    """
    s = err_str.lower()

    # Indices clairs d'epuisement journalier / total
    rpd_keywords = (
        "per_day", "per day", "daily", "quota_exceeded",
        "requests per day", "generativelanguage",
    )
    if any(k in s for k in rpd_keywords):
        return "rpd"

    # resource_exhausted sans mention de "minute" = quota journalier
    if "resource_exhausted" in s and "minute" not in s:
        return "rpd"

    # Indices de limite par minute
    rpm_keywords = ("per_minute", "per minute", "requests per minute", " rpm")
    if any(k in s for k in rpm_keywords):
        return "rpm"

    # Par defaut : traiter comme RPM (attente prudente, moins de perte de temps
    # si c'est en fait un RPD car on cascadera apres les retries)
    return "rpm"


def _gemini_cache_path(video_path: str, cache_dir: str) -> str:
    """Chemin du fichier de cache JSON Gemini pour une video donnee.

    Le hash est base sur le premier Mo de la video (rapide, suffisamment unique).
    """
    with open(video_path, "rb") as f:
        chunk = f.read(1024 * 1024)
    h = hashlib.md5(chunk).hexdigest()[:10]
    stem = Path(video_path).stem
    return os.path.join(cache_dir, f"gemini_cache_{stem}_{h}.json")


def analyze_video_gemini(
    video_path: str,
    api_keys,
    max_retries_per_model: int = 3,
    start_model: int = 0,
    cache_dir: str = "",
) -> dict:
    """
    Envoie la video a Gemini et retourne le JSON d'analyse.

    Consommation de requetes selon la taille de la video :
      - Cache hit         : 0 requete
      - Inline (< 15 MB) : 1 requete  (generate_content uniquement)
      - File API (>= 15M): 3 requetes (upload + generate + delete)
        + polling si le traitement video prend > 5s (rare)

    Strategies anti-429 :
      1. Cache JSON : 0 requete si la meme video a deja ete analysee
      2. Mode inline : 1 requete au lieu de 7-10 pour les petites videos
      3. Cascade flash-lite → flash → pro : part du modele le plus permissif
      4. Classification RPM vs RPD :
           - RPD → skip immediat vers modele suivant
           - RPM → attente 65s+ ou rotation de cle
      5. Rotation multi-cles : bascule sur la cle suivante a chaque 429

    Args:
        video_path  : chemin de la video .mp4
        api_keys    : str (cle unique ou virgule-separees) ou list[str]
        start_model : index de depart dans _GEMINI_FAMILIES (0 = flash-lite)
        cache_dir   : dossier de cache JSON (vide = desactive)
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("[U-ALPHA][Gemini] ERREUR : google-genai non installe.")
        print("  Installer avec : pip install -U google-genai")
        sys.exit(1)

    # Normaliser api_keys en liste
    if isinstance(api_keys, str):
        keys = [k.strip() for k in api_keys.split(",") if k.strip()]
    else:
        keys = [k.strip() for k in api_keys if k and k.strip()]

    if not keys:
        print("[U-ALPHA][Gemini] ERREUR : aucune cle API fournie.")
        sys.exit(1)

    key_count = len(keys)
    if key_count > 1:
        print(f"[U-ALPHA][Gemini] {key_count} cles API — rotation automatique activee")

    # ── Cache : 0 requete si la meme video a deja ete analysee ───────────────
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _gemini_cache_path(video_path, cache_dir)
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"[U-ALPHA][Gemini] Cache hit — 0 requete consommee")
                print(f"  (Supprimer {cache_file} pour forcer une re-analyse)")
                return cached
            except Exception as e:
                print(f"[U-ALPHA][Gemini] Cache illisible ({e}) — re-analyse en cours")
    else:
        cache_file = ""

    key_idx = 0

    def _make_client():
        return genai.Client(api_key=keys[key_idx % key_count])

    client = _make_client()

    # ── Cascade : noms directs, sans appel ListModels (economise 1 requete) ──
    cascade = _GEMINI_FAMILIES[start_model:]
    print(f"[U-ALPHA][Gemini] Cascade : {' → '.join(cascade)}")

    # ── Strategie selon taille : inline (1 req) ou File API (3 req) ──────────
    size_mb = os.path.getsize(video_path) / 1024 / 1024
    use_inline = size_mb <= _MAX_INLINE_MB

    if use_inline:
        print(f"[U-ALPHA][Gemini] {size_mb:.1f} MB → mode inline (1 seule requete)")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_ref = genai_types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")

        def _cleanup():
            pass  # rien a nettoyer en mode inline

    else:
        print(f"[U-ALPHA][Gemini] {size_mb:.1f} MB → mode File API (upload + generate + delete)")
        print("[U-ALPHA][Gemini] Upload en cours...")
        video_file = client.files.upload(
            file=video_path,
            config=genai_types.UploadFileConfig(mime_type="video/mp4")
        )
        poll = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(5)
            poll += 1
            if poll % 6 == 0:
                print(f"[U-ALPHA][Gemini] Traitement Google ({poll*5}s)...")
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            print("[U-ALPHA][Gemini] ERREUR : echec traitement video Google.")
            sys.exit(1)

        video_ref = video_file

        def _cleanup():
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass

    # ── Boucle cascade + retries ─────────────────────────────────────────────
    last_error = None
    gen_cfg = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=GEMINI_RESPONSE_SCHEMA,
        temperature=0.2,
        max_output_tokens=8192,
    )

    for cascade_idx, model_id in enumerate(cascade):
        print(f"[U-ALPHA][Gemini] Modele {cascade_idx+1}/{len(cascade)} : {model_id}")

        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[video_ref, GEMINI_PROMPT],
                    config=gen_cfg,
                )

                raw = response.text.strip() if response.text else ""
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                result = json.loads(raw)

                if cache_file:
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        print(f"[U-ALPHA][Gemini] Cache ecrit : {cache_file}")
                    except Exception as e:
                        print(f"[U-ALPHA][Gemini] Avertissement cache ({e})")

                _cleanup()
                mode_str = "inline" if use_inline else "File API"
                print(f"[U-ALPHA][Gemini] Succes — {model_id} ({mode_str})")
                return result

            except json.JSONDecodeError as e:
                print(f"[U-ALPHA][Gemini] Erreur JSON (tentative {attempt+1}) : {e}")
                print(f"  Reponse brute : {raw[:300]}")
                if attempt < max_retries_per_model - 1:
                    time.sleep(5)
                last_error = e

            except Exception as e:
                err_str = str(e)
                last_error = e
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str.lower()
                    or "rate_limit" in err_str.lower()
                    or "quota" in err_str.lower()
                )

                if is_rate_limit:
                    quota_type = _classify_429(err_str)

                    if quota_type == "rpd":
                        if key_count > 1 and (key_idx + 1) % key_count != 0:
                            key_idx += 1
                            client = _make_client()
                            print(f"[U-ALPHA][Gemini] RPD — rotation vers cle {(key_idx % key_count)+1}/{key_count}")
                            break
                        else:
                            print(f"[U-ALPHA][Gemini] RPD epuise sur {model_id} → modele suivant")
                            break

                    else:
                        if key_count > 1:
                            key_idx += 1
                            client = _make_client()
                            print(f"[U-ALPHA][Gemini] RPM — rotation vers cle {(key_idx % key_count)+1}/{key_count}")
                        else:
                            wait = min(_RPM_WAIT_BASE + attempt * 15, 120)
                            if attempt < max_retries_per_model - 1:
                                print(f"[U-ALPHA][Gemini] RPM — attente {wait}s "
                                      f"(tentative {attempt+1}/{max_retries_per_model})")
                                time.sleep(wait)
                            else:
                                print(f"[U-ALPHA][Gemini] RPM — {max_retries_per_model} tentatives "
                                      f"epuisees sur {model_id} → modele suivant")
                                break

                else:
                    # 404 = modele inexistant ou depreque → skip immediat
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        print(f"[U-ALPHA][Gemini] 404 sur {model_id} → modele suivant")
                        break
                    wait = (attempt + 1) * 3
                    print(f"[U-ALPHA][Gemini] Erreur tentative {attempt+1} "
                          f"({err_str[:120]}) — attente {wait}s")
                    if attempt < max_retries_per_model - 1:
                        time.sleep(wait)

    _cleanup()

    print(f"[U-ALPHA][Gemini] ECHEC — tous les modeles epuises.")
    print(f"  Modeles essayes : {', '.join(cascade)}")
    if key_count == 1:
        print(f"  Conseil : --gemini-keys cle1,cle2 (rotation auto sur 429)")
    print(f"  Quota : https://aistudio.google.com/rate-limit")
    print(f"  ── Tier 1 gratuit (200x quota) : https://console.cloud.google.com/billing ──")
    print(f"  Derniere erreur : {last_error}")
    sys.exit(1)


def filter_segments(gemini_data: dict, min_quality: float = 0.6) -> list:
    """Filtre les segments selon la qualité et le routing modele.

    Logique de routing :
    - modele_recommande "WHAM"            → inclus si qualite >= min_quality
    - modele_recommande "FrankMocap_upper" → inclus (qualite seuil 0.6), marqué WARN (non implémenté)
    - modele_recommande "DECA"             → inclus (qualite seuil 0.5), marqué WARN (non implémenté)
    - modele_recommande "skip" / aucune    → exclu
    - Ancien format sans modele_recommande → compatibilité descendante via corps_visible
    """
    segments = []
    for person in gemini_data.get("persons", []):
        pid = person["person_id"]
        for seg in person.get("segments_valides", []):
            qual = seg["qualite_estimee"]
            corps = seg.get("corps_visible", "")
            modele = seg.get("modele_recommande", "")
            extraction = seg.get("extraction_possible", "")

            # ── Routing nouveau format (modele_recommande présent) ──────────
            if modele:
                if modele == "skip" or extraction == "aucune":
                    print(f"  [SKIP]   P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"modele=skip → exclu")
                    continue
                if modele == "WHAM":
                    if qual < min_quality:
                        print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                              f"qualité {qual:.2f} < {min_quality} → exclu")
                        continue
                elif modele == "FrankMocap_upper":
                    if qual < min_quality:
                        print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                              f"qualité {qual:.2f} < {min_quality} → exclu")
                        continue
                    print(f"  [WARN]   P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"FrankMocap_upper non implémenté — segment inclus, WHAM utilisé en fallback")
                elif modele == "DECA":
                    if qual < 0.5:
                        print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                              f"qualité {qual:.2f} < 0.5 → exclu")
                        continue
                    print(f"  [WARN]   P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"DECA non implémenté — segment marqué, ignoré par WHAM")
                    seg = {**seg, "_skip_wham": True}

            # ── Compatibilité descendante (ancien format sans modele_recommande) ──
            else:
                if qual < min_quality:
                    print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"qualité {qual:.2f} < {min_quality} → exclu")
                    continue
                if corps == "tete_seulement":
                    print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"tete_seulement (ancien format) → exclu")
                    continue
                if corps in ("partiel", "tronc"):
                    print(f"  [WARN]   P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                          f"corps partiel — WHAM peut halluciner les membres manquants")

            segments.append({"person_id": pid, **seg})
    return segments


def print_gemini_report(gemini_data: dict) -> None:
    """Affiche un résumé lisible du rapport Gemini."""
    print("\n" + "=" * 60)
    print("[U-ALPHA][Gemini] RAPPORT D'ANALYSE")
    print("=" * 60)
    print(f"  Durée         : {gemini_data.get('video_duration_seconds', '?')}s")
    print(f"  FPS source    : {gemini_data.get('source_fps', '?')}")
    print(f"  Personnes     : {gemini_data.get('total_persons', '?')}")
    print(f"  Qualité       : {gemini_data.get('qualite_globale', '?')}")
    cam = gemini_data.get("camera", {})
    zoom_warn = " [ZOOM DÉTECTÉ — root motion peut être faussé]" if cam.get("zoom_detecte") else ""
    print(f"  Caméra        : {cam.get('mouvement', '?')}{zoom_warn}")
    if cam.get("mouvement") == "agitee":
        print("  [WARN] Caméra agitée — root motion peu fiable")
    print(f"\n  Recommandation : {gemini_data.get('recommandation', '')}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 2 — EXTRACTION WHAM
# ──────────────────────────────────────────────────────────────────────────────

def cut_video_segment(video_path: str, start_s: float, end_s: float, out_path: str) -> None:
    """Découpe un segment vidéo avec ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", str(start_s),
        "-to", str(end_s),
        "-c", "copy", out_path
    ], check=True)


def _extract_wham_poses(data: dict):
    """Extrait les poses SMPL axis-angle depuis le dict WHAM. Gère plusieurs formats."""
    if "smpl" in data and "poses" in data["smpl"]:
        return np.array(data["smpl"]["poses"]).reshape(-1, 24, 3)
    if "poses" in data:
        p = np.array(data["poses"])
        if p.ndim == 2 and p.shape[1] == 72:
            return p.reshape(-1, 24, 3)
        if p.ndim == 3 and p.shape[1:] == (24, 3):
            return p
    if "pose_body" in data and "root_orient" in data:
        root = np.array(data["root_orient"]).reshape(-1, 1, 3)
        body = np.array(data["pose_body"]).reshape(-1, 23, 3)
        return np.concatenate([root, body], axis=1)
    return None


def _extract_wham_transl(data: dict, n_frames: int) -> np.ndarray:
    """Extrait la translation depuis le dict WHAM."""
    for key in ["trans", "transl", "translation"]:
        if key in data:
            return np.array(data[key])
        if "smpl" in data and key in data["smpl"]:
            return np.array(data["smpl"][key])
    return np.zeros((n_frames, 3))


def run_wham(video_path: str, segments: list, wham_dir: str, tmp_dir: str) -> dict:
    """
    Appelle WHAM sur chaque segment validé et retourne les poses SMPL.
    Retourne : {person_id: {"poses": {frame_idx: (24,3)}, "transl": {frame_idx: (3,)}}}
    """
    wham_dir = Path(wham_dir).expanduser()
    if not wham_dir.exists():
        print(f"[U-ALPHA][WHAM] ERREUR : dépôt WHAM introuvable → {wham_dir}")
        print("  Installer avec :")
        print("    git clone https://github.com/yohanshin/WHAM.git ~/WHAM")
        print("    cd ~/WHAM && pip install -r requirements.txt")
        sys.exit(1)

    demo_script = wham_dir / "demo.py"
    if not demo_script.exists():
        print(f"[U-ALPHA][WHAM] ERREUR : demo.py introuvable dans {wham_dir}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    all_tracks = {}

    for seg in segments:
        pid = seg["person_id"]
        start_s, end_s = seg["start_s"], seg["end_s"]
        print(f"[U-ALPHA][WHAM] P{pid} — segment {start_s:.1f}s-{end_s:.1f}s")

        seg_video = os.path.join(tmp_dir, f"seg_P{pid}_{int(start_s*10):06d}.mp4")
        cut_video_segment(video_path, start_s, end_s, seg_video)

        wham_out = os.path.join(tmp_dir, f"wham_P{pid}_{int(start_s*10):06d}")
        os.makedirs(wham_out, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(demo_script),
             "--video", seg_video,
             "--output_pth", wham_out,
             "--visualize", "false"],
            cwd=str(wham_dir),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[U-ALPHA][WHAM] ERREUR segment P{pid} {start_s:.1f}s : {result.stderr[-300:]}")
            continue

        pkl_files = list(Path(wham_out).glob("*.pkl"))
        if not pkl_files:
            print(f"[U-ALPHA][WHAM] Aucun .pkl dans {wham_out} — segment ignoré")
            continue

        frame_offset = int(start_s * src_fps)

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                wham_data = pickle.load(f)

            poses = _extract_wham_poses(wham_data)
            if poses is None:
                print(f"  [WARN] Format WHAM inconnu dans {pkl_file.name}")
                continue

            transl = _extract_wham_transl(wham_data, len(poses))

            if pid not in all_tracks:
                all_tracks[pid] = {"poses": {}, "transl": {}}

            for t in range(len(poses)):
                all_tracks[pid]["poses"][frame_offset + t] = poses[t]
                all_tracks[pid]["transl"][frame_offset + t] = transl[t]

        n = len(all_tracks.get(pid, {}).get("poses", {}))
        print(f"  OK — {n} frames accumulées pour P{pid}")

    return all_tracks


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 3 — RETARGETING SMPL → R15
# ──────────────────────────────────────────────────────────────────────────────

def smpl_to_r15(poses_aa: np.ndarray, transl: np.ndarray) -> tuple:
    """
    Convertit les poses SMPL axis-angle → quaternions WXYZ R15 Roblox.

    Args:
        poses_aa : (N, 24, 3) axis-angle SMPL
        transl   : (N, 3) translation monde SMPL

    Returns:
        rotations    : (N, 15, 4) quaternions WXYZ float32
        root_position: (N, 3) float32
    """
    n = len(poses_aa)
    rotations = np.zeros((n, 15, 4), dtype=np.float32)
    rotations[:, :, 0] = 1.0  # identité w=1 par défaut

    for smpl_idx, bone_name in SMPL_TO_R15.items():
        r15_idx = R15_INDEX[bone_name]
        aa = poses_aa[:, smpl_idx, :]  # (N, 3)

        # Appliquer convention axes Roblox : X=-X, Z=-Z
        aa_roblox = aa * np.array([-1.0, 1.0, -1.0])

        rot = R.from_rotvec(aa_roblox)
        q_xyzw = rot.as_quat()  # scipy retourne xyzw

        # Convertir xyzw → wxyz (convention Roblox/MOTUS)
        q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=-1)
        rotations[:, r15_idx, :] = q_wxyz.astype(np.float32)

    # Translation : appliquer axes Roblox
    root_position = transl * np.array([-1.0, 1.0, -1.0])
    return rotations, root_position.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 4 — LISSAGE ET INTERPOLATION
# ──────────────────────────────────────────────────────────────────────────────

def smooth_array(data: np.ndarray, window: int, poly: int) -> np.ndarray:
    if len(data) < window:
        return data
    out = np.zeros_like(data)
    for i in range(data.shape[-1]):
        out[..., i] = savgol_filter(data[..., i], window, poly, axis=0)
    return out


def normalize_quats(rots: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rots, axis=-1, keepdims=True)
    return rots / (norms + 1e-12)


def smooth_extremities(rots: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    """Lissage renforcé sur les os extrémités (mains, pieds, avant-bras)."""
    if window % 2 == 0:
        window += 1
    window = max(window, poly + 2)
    if len(rots) < window:
        return rots
    rots_flat = rots.reshape(len(rots), -1).copy()
    for bi in EXTREMITY_INDICES:
        s, e = bi * 4, bi * 4 + 4
        rots_flat[:, s:e] = savgol_filter(rots_flat[:, s:e], window, poly, axis=0)
    result = rots_flat.reshape(-1, len(BONE_NAMES), 4)
    return normalize_quats(result)


def temporal_resample(data: np.ndarray, src_fps: float, tgt_fps: int) -> np.ndarray:
    if src_fps == tgt_fps or len(data) < 2:
        return data
    n_src = len(data)
    n_tgt = int(round(n_src * tgt_fps / src_fps))
    if n_tgt < 2:
        return data
    t_s = np.linspace(0, 1, n_src)
    t_t = np.linspace(0, 1, n_tgt)
    shape = data.shape[1:]
    flat = data.reshape(n_src, -1)
    kind = "cubic" if n_src >= 4 else "linear"
    return interp1d(t_s, flat, axis=0, kind=kind)(t_t).reshape(n_tgt, *shape)


def fill_gaps(poses_dict: dict, transl_dict: dict, max_gap: int = 10) -> None:
    """Interpolation linéaire des trous d'occlusion (max_gap frames)."""
    indices = sorted(poses_dict.keys())
    if len(indices) < 2:
        return
    for a, b in zip(indices, indices[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap + 1:
            continue
        for t in range(a + 1, b):
            alpha = (t - a) / (b - a)
            poses_dict[t] = (1 - alpha) * poses_dict[a] + alpha * poses_dict[b]
            transl_dict[t] = (1 - alpha) * transl_dict[a] + alpha * transl_dict[b]


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 5 — EXPORT .npz
# ──────────────────────────────────────────────────────────────────────────────

def export_npz(rots: np.ndarray, root_pos: np.ndarray, fps: int,
               src_fps: float, duration: float, person_index: int,
               total_persons: int, out_path: str) -> None:
    np.savez(
        out_path,
        rotations=rots.astype(np.float32),
        root_position=root_pos.astype(np.float32),
        bone_names=np.array(BONE_NAMES),
        fps=np.int32(fps),
        duration=np.float64(duration),
        source_fps=np.int32(int(src_fps)),
        person_index=np.int32(person_index),
        total_persons=np.int32(total_persons),
    )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ANIMA-MECHANICUS — Frégate U-ALPHA : L'Auspex Cogitateur"
    )
    parser.add_argument("input", help="Chemin vidéo .mp4 (humain réel uniquement)")

    # Cles Gemini — deux formes acceptées pour la compatibilite
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--gemini-key",
                           help="Cle API Gemini unique (alias de --gemini-keys)")
    key_group.add_argument("--gemini-keys",
                           help="Cles API Gemini separees par virgule (rotation auto sur 429). "
                                "Ex: --gemini-keys cle1,cle2,cle3")

    parser.add_argument("-o", "--output", default="outputs/")
    parser.add_argument("--fps", type=int, default=30, choices=[30, 60, 120])
    parser.add_argument("--smooth", default="moyen",
                        choices=["faible", "moyen", "brutal"])
    parser.add_argument("--no-root-motion", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--wham-dir", default="~/WHAM")
    parser.add_argument("--cache-dir", default="",
                        help="Dossier pour cacher les resultats Gemini (evite re-analyse). "
                             "Ex: --cache-dir outputs/gemini_cache")
    parser.add_argument("--start-model", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Index de depart dans la cascade Gemini "
                             "(0=pro, 1=2.5flash, 2=2.0flash, 3=1.5flash). "
                             "Utile si le pro est connu comme epuise.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERREUR] Fichier introuvable : {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {args.input}")
        sys.exit(1)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    cap.release()
    print(f"[U-ALPHA] Vidéo : {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")

    # Normaliser les cles
    api_keys = args.gemini_keys if args.gemini_keys else args.gemini_key

    # ── Etape 1 : Analyse Gemini ─────────────────────────────────────────────
    print("\n[U-ALPHA] Etape 1/4 — Analyse Gemini...")
    gemini_data = analyze_video_gemini(
        args.input,
        api_keys,
        start_model=args.start_model,
        cache_dir=args.cache_dir,
    )
    print_gemini_report(gemini_data)

    segments = filter_segments(gemini_data, args.quality_threshold)
    if not segments:
        print("[ERREUR] Aucun segment valide après filtrage Gemini.")
        print("  Vérifier que la vidéo contient un humain réel clairement visible.")
        sys.exit(1)
    print(f"[U-ALPHA] {len(segments)} segment(s) retenu(s) pour WHAM")

    # ── Etape 2 : Extraction WHAM ────────────────────────────────────────────
    print("\n[U-ALPHA] Etape 2/4 — Extraction WHAM (poses SMPL)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        wham_tracks = run_wham(args.input, segments, args.wham_dir, tmp_dir)

    if not wham_tracks:
        print("[ERREUR] WHAM n'a produit aucun résultat.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    n_persons = len(wham_tracks)
    win, poly = SMOOTH_PRESETS[args.smooth]
    print(f"\n[U-ALPHA] Etapes 3-4/4 — SMPL→R15 + Lissage + Export ({n_persons} personne(s))...")

    # ── Etapes 3-4 : Retargeting + Lissage + Export ──────────────────────────
    for pid in sorted(wham_tracks.keys()):
        track = wham_tracks[pid]
        poses_dict = track["poses"]
        transl_dict = track["transl"]

        fill_gaps(poses_dict, transl_dict)

        indices = sorted(poses_dict.keys())
        poses_aa = np.array([poses_dict[i] for i in indices])  # (N, 24, 3)
        transl = np.array([transl_dict[i] for i in indices])   # (N, 3)

        # Retargeting SMPL → R15
        rots, root_pos = smpl_to_r15(poses_aa, transl)

        if args.no_root_motion:
            root_pos = np.zeros_like(root_pos)

        # Lissage global
        root_pos = smooth_array(root_pos, win, poly)
        rots = smooth_array(rots.reshape(len(rots), -1), win, poly).reshape(-1, 15, 4)
        rots = normalize_quats(rots)

        # Lissage renforcé extrémités
        ext_win = max(win * 2 + 1, 15)
        rots = smooth_extremities(rots, ext_win)

        # Resampling FPS
        root_pos = temporal_resample(root_pos, src_fps, args.fps)
        rots = temporal_resample(
            rots.reshape(len(rots), -1), src_fps, args.fps
        ).reshape(-1, 15, 4)
        rots = normalize_quats(rots)

        out_path = os.path.join(args.output, f"motus_core_P{pid}.npz")
        export_npz(rots, root_pos, args.fps, src_fps, duration, pid, n_persons, out_path)
        print(f"  → {out_path} ({rots.shape[0]} frames @ {args.fps} FPS)")

    print(f"\n[U-ALPHA] Extraction terminée — {n_persons} fichier(s) .npz exporté(s)")
    print("[U-ALPHA] Prêt pour la Frégate U-GAMMA")


if __name__ == "__main__":
    main()


