#!/usr/bin/env python3
"""MOTUS-VIGILUS — Frégate U-ALPHA : L'Auspex — Extraction de mouvement vidéo → .npz R15"""

import sys, os, argparse, urllib.request
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scenedetect import detect, ContentDetector

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task")

BONE_NAMES = [
    "LowerTorso", "UpperTorso", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]

BONE_DEFS = [
    ("LowerTorso",    "mid_23_24", "mid_11_12"),
    ("UpperTorso",    "mid_11_12", "upper_neck"),
    ("Head",          "mid_9_10",  "pt_0"),
    ("LeftUpperArm",  "pt_11", "pt_13"), ("LeftLowerArm",  "pt_13", "pt_15"),
    ("LeftHand",      "pt_15", "pt_19"),
    ("RightUpperArm", "pt_12", "pt_14"), ("RightLowerArm", "pt_14", "pt_16"),
    ("RightHand",     "pt_16", "pt_20"),
    ("LeftUpperLeg",  "pt_23", "pt_25"), ("LeftLowerLeg",  "pt_25", "pt_27"),
    ("LeftFoot",      "pt_27", "pt_31"),
    ("RightUpperLeg", "pt_24", "pt_26"), ("RightLowerLeg", "pt_26", "pt_28"),
    ("RightFoot",     "pt_28", "pt_32"),
]

REST_VECTORS = {
    "LowerTorso":    [ 0.000,  1.000,  0.000],
    "UpperTorso":    [ 0.000,  1.000,  0.000],
    "Head":          [ 0.000,  1.000,  0.000],
    "LeftUpperArm":  [-0.589, -0.808,  0.000],
    "LeftLowerArm":  [-0.547, -0.809,  0.213],
    "LeftHand":      [-0.547, -0.809,  0.213],
    "RightUpperArm": [ 0.589, -0.808,  0.000],
    "RightLowerArm": [ 0.547, -0.809,  0.213],
    "RightHand":     [ 0.547, -0.809,  0.213],
    "LeftUpperLeg":  [-0.137, -0.990, -0.032],
    "LeftLowerLeg":  [-0.118, -0.990, -0.072],
    "LeftFoot":      [ 0.000,  0.000,  1.000],
    "RightUpperLeg": [ 0.137, -0.990, -0.032],
    "RightLowerLeg": [ 0.118, -0.990, -0.072],
    "RightFoot":     [ 0.000,  0.000,  1.000],
}

SMOOTH_PRESETS = {"faible": (5, 2), "moyen": (7, 3), "brutal": (15, 3)}

# Bones extremites sujets aux artefacts MediaPipe — lissage renforce systematiquement
EXTREMITY_BONE_NAMES = {
    "LeftLowerArm", "LeftHand",
    "RightLowerArm", "RightHand",
    "LeftLowerLeg", "LeftFoot",
    "RightLowerLeg", "RightFoot",
}
EXTREMITY_INDICES = [i for i, name in enumerate(BONE_NAMES) if name in EXTREMITY_BONE_NAMES]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[U-ALPHA] Téléchargement du modèle → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def axis_map(v):
    return np.array([-v[0], v[1], -v[2]])


def resolve_point(lm, tag):
    if tag == "mid_23_24":  return (lm[23] + lm[24]) / 2
    if tag == "mid_11_12":  return (lm[11] + lm[12]) / 2
    if tag == "mid_9_10":   return (lm[9] + lm[10]) / 2
    if tag == "upper_neck":
        mid_sh = (lm[11] + lm[12]) / 2
        return mid_sh + np.array([0, 0.15, 0])
    return lm[int(tag.split("_")[1])]


def vec2quat(v_from, v_to):
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-12)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if dot < -0.9999:
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    w = 1.0 + dot
    q = np.array([w, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


def landmarks_to_rotations(lm_mapped):
    rots = np.zeros((len(BONE_DEFS), 4))
    for i, (name, src, dst) in enumerate(BONE_DEFS):
        p0, p1 = resolve_point(lm_mapped, src), resolve_point(lm_mapped, dst)
        bone_vec = p1 - p0
        if np.linalg.norm(bone_vec) < 1e-8:
            rots[i] = [1, 0, 0, 0]
            continue
        rest = np.array(REST_VECTORS[name], dtype=np.float64)
        rots[i] = vec2quat(rest, bone_vec)
    return rots


def smooth_array(data, window, poly, is_quat=False):
    if len(data) < window:
        return data
    out = np.zeros_like(data)
    for i in range(data.shape[-1]):
        out[..., i] = savgol_filter(data[..., i], window, poly, axis=0)
    if is_quat:
        norms = np.linalg.norm(out.reshape(-1, 4), axis=1, keepdims=True)
        out = (out.reshape(-1, 4) / (norms + 1e-12)).reshape(out.shape)
    return out


def smooth_extremities(rots, window, poly=3):
    """Passe de lissage renforcee sur les bones extremites (pieds, mains, avant-bras)."""
    # window doit etre impair et >= poly+2
    if window % 2 == 0:
        window += 1
    window = max(window, poly + 2)
    if len(rots) < window:
        return rots
    rots_flat = rots.reshape(len(rots), -1).copy()
    for bi in EXTREMITY_INDICES:
        col_s, col_e = bi * 4, bi * 4 + 4
        rots_flat[:, col_s:col_e] = savgol_filter(
            rots_flat[:, col_s:col_e], window, poly, axis=0
        )
    result = rots_flat.reshape(-1, len(BONE_NAMES), 4)
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    return result / (norms + 1e-12)


def temporal_resample(data, src_fps, tgt_fps):
    """Ré-échantillonnage temporel : gère upscaling ET downscaling via interpolation cubique."""
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


def fill_gaps(tracks, max_gap=10):
    for pid in tracks:
        frames = tracks[pid]
        indices = sorted(frames.keys())
        if len(indices) < 2:
            continue
        for a, b in zip(indices, indices[1:]):
            gap = b - a
            if gap <= 1 or gap > max_gap + 1:
                continue
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                frames[t] = (1 - alpha) * frames[a] + alpha * frames[b]


def extract_scene(cap, start_f, end_f, landmarker, src_fps, tracks):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for fi in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(fi * 1000 / src_fps)
        result = landmarker.detect_for_video(mp_img, ts)
        n_people = len(result.pose_world_landmarks)
        if fi % 50 == 0:
            print(f"  Frame {fi}/{end_f} — {n_people} personne(s)")
        for pid, wl in enumerate(result.pose_world_landmarks):
            lm = np.array([[l.x, l.y, l.z] for l in wl])
            lm_mapped = np.array([axis_map(v) for v in lm])
            root = (lm_mapped[23] + lm_mapped[24]) / 2
            rots = landmarks_to_rotations(lm_mapped)
            tracks.setdefault(pid, {})[fi] = np.concatenate([root, rots.flatten()])


def main():
    parser = argparse.ArgumentParser(description="MOTUS-VIGILUS — Frégate U-ALPHA : L'Auspex")
    parser.add_argument("input", help="Chemin vidéo .mp4")
    parser.add_argument("-o", "--output", default="outputs/", help="Dossier de sortie")
    parser.add_argument("--fps", type=int, default=30, choices=[30, 60, 120], help="FPS cible")
    parser.add_argument("--smooth", default="moyen", choices=["faible", "moyen", "brutal"])
    parser.add_argument("--no-root-motion", action="store_true", help="Désactive root motion")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERREUR] Fichier introuvable : {args.input}"); sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la vidéo : {args.input}"); sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    print(f"[U-ALPHA] Vidéo : {total_frames} frames, {src_fps:.1f} FPS, {duration:.1f}s")

    print("[U-ALPHA] Détection de scènes...")
    scenes = detect(args.input, ContentDetector(threshold=27.0))
    if not scenes:
        scenes = [(0, total_frames)]
    else:
        scenes = [(s[0].get_frames(), s[1].get_frames()) for s in scenes]
    print(f"  {len(scenes)} scène(s) détectée(s)")

    ensure_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    tracks = {}
    print("[U-ALPHA] Extraction des poses...")
    for si, (sf, ef) in enumerate(scenes):
        print(f"  Scène {si + 1}/{len(scenes)} [{sf}→{ef}]")
        extract_scene(cap, sf, ef, landmarker, src_fps, tracks)
    cap.release()
    landmarker.close()

    if not tracks:
        print("[ERREUR] Aucune personne détectée dans la vidéo."); sys.exit(1)

    fill_gaps(tracks)
    win, poly = SMOOTH_PRESETS[args.smooth]
    os.makedirs(args.output, exist_ok=True)
    n_persons = len(tracks)
    print(f"[U-ALPHA] {n_persons} personne(s) — lissage '{args.smooth}', cible {args.fps} FPS")

    for pid in sorted(tracks.keys()):
        frames = tracks[pid]
        indices = sorted(frames.keys())
        raw = np.array([frames[i] for i in indices])
        root_pos = raw[:, :3]
        rots = raw[:, 3:].reshape(len(raw), 15, 4)

        if args.no_root_motion:
            root_pos = np.zeros_like(root_pos)

        # Lissage principal (tous les bones)
        root_pos = smooth_array(root_pos, win, poly)
        rots = smooth_array(rots.reshape(len(rots), -1), win, poly).reshape(-1, 15, 4)
        norms = np.linalg.norm(rots, axis=-1, keepdims=True)
        rots = rots / (norms + 1e-12)

        # Lissage renforce sur les extremites (window = max(win*2, 15))
        ext_win = max(win * 2 + 1, 15)
        rots = smooth_extremities(rots, ext_win)

        # Ré-échantillonnage temporel (upscaling ET downscaling corrigé)
        root_pos = temporal_resample(root_pos, src_fps, args.fps)
        rots = temporal_resample(rots.reshape(len(rots), -1), src_fps, args.fps).reshape(-1, 15, 4)
        norms = np.linalg.norm(rots, axis=-1, keepdims=True)
        rots = rots / (norms + 1e-12)

        out_path = os.path.join(args.output, f"motus_core_P{pid}.npz")
        np.savez(
            out_path,
            rotations=rots.astype(np.float32),
            root_position=root_pos.astype(np.float32),
            bone_names=np.array(BONE_NAMES),
            fps=np.int32(args.fps),
            duration=np.float64(duration),
            source_fps=np.int32(int(src_fps)),
            person_index=np.int32(pid),
            total_persons=np.int32(n_persons),
        )
        print(f"  → {out_path} ({rots.shape[0]} frames)")

    print(f"[U-ALPHA] Extraction terminée — {n_persons} fichier(s) .npz exporté(s)")


if __name__ == "__main__":
    main()
