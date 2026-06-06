"""
real_hri.py

Real-robot HRI helpers for LILAC utterance triggering, head targets, ZED RGB,
and face-centered head tracking.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent

HRI_TRIGGER_PORT = 5581
HRI_HEAD_TARGET_PORT = 5582
HRI_STATUS_PORT = 5583
ZED_RGB_PORT = 5999
LOCALHOST_IP = "127.0.0.1"

HEAD_PITCH_JOINT_NAME = "head_joint1"
HEAD_YAW_JOINT_NAME = "head_joint2"
HEAD_PITCH_RANGE_DEG = (-50.0, 30.0)
HEAD_YAW_RANGE_DEG = (-20.0, 20.0)
HEAD_CENTER_DEG = (0.0, 0.0)
HEAD_DESK_DEG = (-45.0, 0.0)

FACE_DB_DIR = PROJECT_DIR / "data" / "face_db"
FACE_MODEL_CACHE_DIR = PROJECT_DIR / "cache" / "hf_face"


def clip_head_deg(pitch_deg, yaw_deg):
    pitch = float(np.clip(float(pitch_deg), *HEAD_PITCH_RANGE_DEG))
    yaw = float(np.clip(float(yaw_deg), *HEAD_YAW_RANGE_DEG))
    return pitch, yaw


def make_hri_trigger_message(source="vader5_button_11", trigger_id=None):
    return {
        "msg_type": "lilac_hri_trigger",
        "time": time.time(),
        "source": str(source),
        "trigger_id": trigger_id or ("trigger_%d" % int(time.time() * 1000)),
    }


def make_head_target_message(
        pitch_deg,
        yaw_deg,
        mode="target",
        source="lilac_hri",
        max_speed_deg_s=180.0,
        trigger_id=None,
    ):
    pitch_deg, yaw_deg = clip_head_deg(pitch_deg, yaw_deg)
    return {
        "msg_type": "lilac_head_target",
        "time": time.time(),
        "mode": str(mode),
        "source": str(source),
        "trigger_id": trigger_id,
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
        "max_speed_deg_s": float(max_speed_deg_s),
    }


def make_center_head_target(**kwargs):
    return make_head_target_message(
        pitch_deg=HEAD_CENTER_DEG[0],
        yaw_deg=HEAD_CENTER_DEG[1],
        mode="center",
        **kwargs,
    )


def make_desk_head_target(**kwargs):
    return make_head_target_message(
        pitch_deg=HEAD_DESK_DEG[0],
        yaw_deg=HEAD_DESK_DEG[1],
        mode="desk",
        **kwargs,
    )


def parse_head_target_message(msg):
    if not isinstance(msg, dict):
        return None
    if msg.get("msg_type") != "lilac_head_target":
        return None

    pitch_deg, yaw_deg = clip_head_deg(
        msg.get("pitch_deg", HEAD_CENTER_DEG[0]),
        msg.get("yaw_deg", HEAD_CENTER_DEG[1]),
    )
    out = dict(msg)
    out["pitch_deg"] = pitch_deg
    out["yaw_deg"] = yaw_deg
    out["pitch_rad"] = float(np.deg2rad(pitch_deg))
    out["yaw_rad"] = float(np.deg2rad(yaw_deg))
    out["max_speed_rad_s"] = float(np.deg2rad(float(msg.get("max_speed_deg_s", 180.0))))
    return out


def get_head_joint_indices(q_msg_layout):
    use_joint_names = list(q_msg_layout["use_joint_names"])
    return {
        "pitch": use_joint_names.index(HEAD_PITCH_JOINT_NAME),
        "yaw": use_joint_names.index(HEAD_YAW_JOINT_NAME),
    }


def step_scalar_towards(current, target, max_delta):
    current = float(current)
    target = float(target)
    max_delta = abs(float(max_delta))
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + np.sign(delta) * max_delta


def step_head_qpos_towards_target(qpos, q_msg_layout, target_msg, dt, default_speed_deg_s=180.0):
    target = parse_head_target_message(target_msg)
    qpos_next = np.asarray(qpos, dtype=np.float64).copy()
    if target is None:
        return qpos_next, {"active": False}

    idxs = get_head_joint_indices(q_msg_layout)
    speed = float(target.get("max_speed_rad_s", np.deg2rad(default_speed_deg_s)))
    max_delta = speed * max(0.0, float(dt))

    qpos_next[idxs["pitch"]] = step_scalar_towards(
        qpos_next[idxs["pitch"]],
        target["pitch_rad"],
        max_delta,
    )
    qpos_next[idxs["yaw"]] = step_scalar_towards(
        qpos_next[idxs["yaw"]],
        target["yaw_rad"],
        max_delta,
    )

    pitch_err = target["pitch_rad"] - qpos_next[idxs["pitch"]]
    yaw_err = target["yaw_rad"] - qpos_next[idxs["yaw"]]
    return qpos_next, {
        "active": True,
        "mode": target.get("mode", "target"),
        "pitch_deg": float(np.rad2deg(qpos_next[idxs["pitch"]])),
        "yaw_deg": float(np.rad2deg(qpos_next[idxs["yaw"]])),
        "target_pitch_deg": target["pitch_deg"],
        "target_yaw_deg": target["yaw_deg"],
        "err_deg": float(np.rad2deg(np.linalg.norm([pitch_err, yaw_err]))),
    }


def is_head_target_reached(qpos, q_msg_layout, target_msg, tol_deg=2.0):
    target = parse_head_target_message(target_msg)
    if target is None:
        return False
    idxs = get_head_joint_indices(q_msg_layout)
    qpos = np.asarray(qpos, dtype=np.float64)
    err = np.linalg.norm([
        target["pitch_rad"] - qpos[idxs["pitch"]],
        target["yaw_rad"] - qpos[idxs["yaw"]],
    ])
    return bool(err <= np.deg2rad(float(tol_deg)))


def build_face_identifier(
        face_db_dir=FACE_DB_DIR,
        model_cache_dir=FACE_MODEL_CACHE_DIR,
        id_threshold=0.4,
        verbose=True,
    ):
    from ri_motion_v5_package.hf_models import FaceEmbedding, FaceIdentification, YoloFaceDet

    model_cache_dir = Path(model_cache_dir)
    face_det = YoloFaceDet(
        ckpt_repo = "AdamCodd/YOLOv11n-face-detection",
        ckpt_file = "model.pt",
        local_dir = str(model_cache_dir / "yolo_face_det"),
        verbose   = verbose,
    )
    face_enc = FaceEmbedding(
        ckpt_repo = "deepghs/insightface",
        ckpt_file = "buffalo_l/w600k_r50.onnx",
        local_dir = str(model_cache_dir / "face_embedding"),
        providers = ["CPUExecutionProvider"],
        verbose   = verbose,
    )
    face_id = FaceIdentification(
        detector     = face_det,
        encoder      = face_enc,
        id_threshold = float(id_threshold),
        verbose      = verbose,
    )
    face_db_dir = Path(face_db_dir)
    face_db_dir.mkdir(parents=True, exist_ok=True)
    face_id.build_db_from_dir(db_dir=str(face_db_dir))
    return face_id


def rgb_to_bgr(image_rgb):
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    return image_rgb[:, :, ::-1].copy()


def normalize_face_name(name):
    name = "" if name is None else str(name).strip()
    stem = Path(name).stem.lower()
    stem = re.sub(r"[\s_-]*\d+$", "", stem)
    return stem


def is_face_name_match(name, target_name):
    target_key = normalize_face_name(target_name)
    if not target_key:
        return False
    return normalize_face_name(name) == target_key


def select_face_box(face_out, target_name=None):
    if not face_out:
        return None

    boxes = np.asarray(face_out.get("face_boxes", []), dtype=np.float64)
    if boxes.size == 0:
        return None

    pred_names = list(face_out.get("pred_names", []))
    nearest_names = list(face_out.get("nearest_names", []))
    target_name = "" if target_name is None else str(target_name).strip()

    if target_name:
        for idx, name in enumerate(pred_names):
            if is_face_name_match(name, target_name):
                return boxes[idx], normalize_face_name(target_name)
        for idx, name in enumerate(nearest_names):
            if is_face_name_match(name, target_name):
                return boxes[idx], normalize_face_name(target_name)
        return None

    for idx, name in enumerate(pred_names):
        if name != "Unknown":
            return boxes[idx], name

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    return boxes[idx], pred_names[idx] if idx < len(pred_names) else "Unknown"


def get_face_debug_rows(face_out, target_name=None):
    if not face_out:
        return []

    boxes = np.asarray(face_out.get("face_boxes", []), dtype=np.float64)
    pred_names = list(face_out.get("pred_names", []))
    pred_sims = list(face_out.get("pred_sims", []))
    nearest_names = list(face_out.get("nearest_names", []))

    rows = []
    for idx, box in enumerate(boxes):
        pred_name = pred_names[idx] if idx < len(pred_names) else "Unknown"
        nearest_name = nearest_names[idx] if idx < len(nearest_names) else ""
        pred_sim = float(pred_sims[idx]) if idx < len(pred_sims) else float("nan")
        is_known = str(pred_name) != "Unknown"
        is_target = (
            is_face_name_match(pred_name, target_name)
            or is_face_name_match(nearest_name, target_name)
        )
        rows.append({
            "idx": idx,
            "bbox": [int(round(v)) for v in box.tolist()],
            "pred_name": str(pred_name),
            "nearest_name": str(nearest_name),
            "pred_sim": pred_sim,
            "is_known": bool(is_known),
            "is_target": bool(is_target),
        })
    return rows


def annotate_face_identification(face_out, target_name=None):
    import cv2

    if not face_out or face_out.get("img_bgr", None) is None:
        return None

    vis = np.asarray(face_out["img_bgr"], dtype=np.uint8).copy()
    h, w = vis.shape[:2]

    for row in get_face_debug_rows(face_out, target_name=target_name):
        x1, y1, x2, y2 = row["bbox"]
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

        if row["is_target"]:
            color = (0, 255, 0)
        elif row["is_known"]:
            color = (255, 180, 0)
        else:
            color = (0, 0, 255)

        label_name = row["pred_name"]
        if label_name == "Unknown" and row["nearest_name"]:
            label_name = "Unknown/%s" % row["nearest_name"]
        label = "%s %.2f" % (label_name, row["pred_sim"])

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y1 - label_size[1] - 10)
        cv2.rectangle(vis, (x1, y0), (min(w - 1, x1 + label_size[0]), y1), color, -1)
        cv2.putText(vis, label, (x1, max(label_size[1], y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis[:, :, ::-1].copy()


def face_center_error(box, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    err_x = (cx - 0.5 * float(w)) / max(1.0, 0.5 * float(w))
    err_y = (cy - 0.5 * float(h)) / max(1.0, 0.5 * float(h))
    return float(err_x), float(err_y)


def is_face_centered(box, image_shape, tol=0.08):
    err_x, err_y = face_center_error(box, image_shape)
    return bool(abs(err_x) <= float(tol) and abs(err_y) <= float(tol))


def head_target_from_face_error(
        current_pitch_deg,
        current_yaw_deg,
        box,
        image_shape,
        pitch_gain_deg=10.0,
        yaw_gain_deg=12.0,
        pitch_sign=-1.0,
        yaw_sign=1.0,
    ):
    err_x, err_y = face_center_error(box, image_shape)
    pitch = float(current_pitch_deg) + float(pitch_sign) * float(pitch_gain_deg) * err_y
    yaw = float(current_yaw_deg) + float(yaw_sign) * float(yaw_gain_deg) * err_x
    pitch, yaw = clip_head_deg(pitch, yaw)
    return pitch, yaw, {"err_x": err_x, "err_y": err_y}
