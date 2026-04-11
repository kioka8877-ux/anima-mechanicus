#!/usr/bin/env python3
"""ANIMA-MECHANICUS — Frégate U-ALPHA : L'Auspex Cogitateur — v2
Pipeline : Gemini 2.0 Flash (analyse) → WHAM (SMPL) → SMPL→R15 → .npz

Usage:
    python motus_extract.py video.mp4 --gemini-key YOUR_KEY [options]

Options:
    --fps {30,60,120}                FPS cible (défaut: 30)
    --smooth {faible,moyen,brutal}   Niveau lissage (défaut: moyen)
    --no-root-motion                 Désactive la translation globale
    --quality-threshold F            Score Gemini minimum (défaut: 0.6)
    --output DIR                     Dossier de sortie (défaut: outputs/)
    --wham-dir DIR                   Chemin vers le dépôt WHAM (défaut: ~/WHAM)
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import pickle
import time
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

GEMINI_PROMPT = """Tu es un analyseur de vidéo expert en motion capture. Analyse cette vidéo et retourne UNIQUEMENT un JSON valide, sans markdown, sans explication.

Format attendu :
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
          "corps_visible": "<complet|partiel|tete_seulement>",
          "orientation": "<face|profil|dos|mixte>",
          "distance_camera": "<proche|moyen|lointain>",
          "type_mouvement": "<marche|danse|combat|sport|statique>",
          "qualite_estimee": <float 0.0-1.0>,
          "problemes": []
        }
      ],
      "segments_exclus": [
        {
          "start_s": <float>,
          "end_s": <float>,
          "raison": "<personne_absente|flou_mouvement|occlusion_totale|contre_jour>"
        }
      ]
    }
  ],
  "camera": {
    "mouvement": "<stable|panoramique|suivi|agitee>",
    "zoom_detecte": <bool>
  },
  "qualite_globale": "<excellente|bonne|moyenne|mauvaise>",
  "recommandation": "<string>"
}

Règles d'évaluation :
- corps_visible "complet" : tête + torse + bras + jambes tous visibles
- corps_visible "partiel" : au moins torse + 1 membre visible
- qualite_estimee 1.0 = corps complet, face caméra, bonne lumière, sans flou
- qualite_estimee 0.6 = minimum utilisable pour motion capture
- qualite_estimee < 0.6 = mettre le segment dans segments_exclus
- zoom_detecte true si la caméra zoome ou dé-zoome visiblement
- Créer un person_id distinct par personne présente dans la vidéo
"""


# ──────────────────────────────────────────────────────────────────────────────
# MODULE 1 — ANALYSE GEMINI
# ──────────────────────────────────────────────────────────────────────────────

def _pick_best_gemini_model(client) -> str:
    """
    Detecte automatiquement le meilleur modele Gemini disponible.
    Priorite : pro > flash, version la plus recente en premier.
    """
    PRIORITY = [
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.0-pro-exp",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    try:
        available = [m.name.replace("models/", "") for m in client.models.list()]
        for candidate in PRIORITY:
            matches = [m for m in available if m == candidate or m.startswith(candidate + "-")]
            if matches:
                chosen = sorted(matches)[-1]
                print(f"[U-ALPHA][Gemini] Modele selectionne : {chosen}")
                return chosen
    except Exception as e:
        print(f"[U-ALPHA][Gemini] Impossible de lister les modeles ({e}) — fallback gemini-1.5-pro")
    return "gemini-1.5-pro"


def analyze_video_gemini(video_path: str, api_key: str, max_retries: int = 5) -> dict:
    """Envoie la video a Gemini (meilleur modele disponible) et retourne le JSON d'analyse."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        print("[U-ALPHA][Gemini] ERREUR : google-genai non installe.")
        print("  Installer avec : pip install -U google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    model_id = _pick_best_gemini_model(client)

    print("[U-ALPHA][Gemini] Upload video en cours...")
    video_file = client.files.upload(
        file=video_path,
        config=genai_types.UploadFileConfig(mime_type="video/mp4")
    )

    while video_file.state.name == "PROCESSING":
        print("[U-ALPHA][Gemini] Traitement video sur les serveurs Google...")
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        print("[U-ALPHA][Gemini] ERREUR : echec du traitement video par Google.")
        sys.exit(1)

    print(f"[U-ALPHA][Gemini] Analyse en cours avec {model_id}...")

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[video_file, GEMINI_PROMPT],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=8192,
                )
            )

            raw = response.text.strip() if response.text else ""
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            client.files.delete(name=video_file.name)
            return result

        except json.JSONDecodeError as e:
            print(f"[U-ALPHA][Gemini] ERREUR parsing JSON (tentative {attempt+1}) : {e}")
            print(f"  Reponse brute : {raw[:300]}")
            if attempt < max_retries - 1:
                time.sleep(5)

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "resource_exhausted" in err_str.lower() or "rate" in err_str.lower()
            if is_rate_limit:
                wait = 65
                print(f"[U-ALPHA][Gemini] Rate limit 429 — attente {wait}s puis retry ({attempt+1}/{max_retries})")
            else:
                wait = (attempt + 1) * 5
                print(f"[U-ALPHA][Gemini] Erreur tentative {attempt+1} : {err_str[:200]}")
            if attempt < max_retries - 1:
                time.sleep(wait)

    print(f"[U-ALPHA][Gemini] ECHEC apres {max_retries} tentatives.")
    sys.exit(1)


def filter_segments(gemini_data: dict, min_quality: float = 0.6) -> list:
    """Filtre les segments selon la qualité et retourne la liste à traiter."""
    segments = []
    for person in gemini_data.get("persons", []):
        pid = person["person_id"]
        for seg in person.get("segments_valides", []):
            if seg["qualite_estimee"] < min_quality:
                print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                      f"qualité {seg['qualite_estimee']:.2f} < {min_quality} → exclu")
                continue
            if seg["corps_visible"] == "tete_seulement":
                print(f"  [FILTRE] P{pid} {seg['start_s']:.1f}s-{seg['end_s']:.1f}s : "
                      f"corps non visible → exclu")
                continue
            if seg["corps_visible"] == "partiel":
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
    parser.add_argument("--gemini-key", required=True,
                        help="Clé API Gemini (gratuite sur aistudio.google.com)")
    parser.add_argument("-o", "--output", default="outputs/")
    parser.add_argument("--fps", type=int, default=30, choices=[30, 60, 120])
    parser.add_argument("--smooth", default="moyen",
                        choices=["faible", "moyen", "brutal"])
    parser.add_argument("--no-root-motion", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--wham-dir", default="~/WHAM")
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

    # ── Etape 1 : Analyse Gemini ─────────────────────────────────────────────
    print("\n[U-ALPHA] Etape 1/4 — Analyse Gemini 2.0 Flash...")
    gemini_data = analyze_video_gemini(args.input, args.gemini_key)
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
