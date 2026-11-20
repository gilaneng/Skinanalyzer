"""
Skin Analyzer – Streamlit app
Run:  streamlit run app.py

Pipeline: selfie upload -> face detection (OpenCV Haar cascade) -> skin mask
-> pigmentation / texture / redness scores -> skin age estimate
-> consultation report -> PDF export -> product recommendations.

NOTE: Scores are image-based heuristics for cosmetic consultation only.
They are NOT a medical diagnosis.
"""

import hashlib
import html
import io
import os
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import cv2
import matplotlib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ----------------------------------------------------------------------------
# Product catalogue  (edit descriptions / targets to match your real products)
# "targets" decide when a product is recommended.
# ----------------------------------------------------------------------------
PRODUCTS = {
    "Active Serum 667": {
        "targets": ["pigmentation", "texture"],
        "role": "Targeted serum step for uneven tone and surface refinement.",
    },
    "Medicare MD Cream 683": {
        "targets": ["redness"],
        "role": "Cream step for comfort and calming of visibly reactive skin.",
    },
    "Vitamin A.C.E Treatment Effector 450": {
        "targets": ["age", "pigmentation", "texture"],
        "role": "Treatment step for visible signs of ageing and overall radiance.",
    },
}

APP_VERSION = "2026-11-20 contour + score comparison (diagnostics)"
CONCERN_THRESHOLD = 35  # score at/above which a concern is flagged


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Analysis:
    pigmentation: float
    texture: float
    redness: float
    skin_age: int
    chronological_age: int
    face_box: tuple
    annotated: np.ndarray  # RGB image with face box + skin mask overlay
    quality_notes: list


LEVEL_COLORS = {"Low": "#3E8E6B", "Mild": "#B58A1B", "Moderate": "#CF6B3F", "High": "#B23A48"}


def label(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Mild"
    if score < 75:
        return "Moderate"
    return "High"


# ----------------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------------
def load_image(file) -> np.ndarray:
    """Load upload as RGB ndarray, honouring EXIF rotation, capped at 1400px."""
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1400, 1400))
    return np.array(img)


def detect_face(rgb: np.ndarray):
    """Return (x, y, w, h) of the largest frontal face, or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError(
            "Face-detection model not found in this OpenCV build. "
            "Install a build that bundles it: pip install 'opencv-python-headless<5'"
        )
    min_side = max(60, int(min(gray.shape) * 0.15))
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side)
    )
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def skin_mask(face_rgb: np.ndarray) -> np.ndarray:
    """Binary mask of skin pixels inside the face crop (YCrCb + central ellipse)."""
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 180, 135))

    h, w = mask.shape
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, (w // 2, int(h * 0.52)), (int(w * 0.38), int(h * 0.46)),
                0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, ellipse)

    # Drop the eye band and mouth zone (high-contrast, not skin quality)
    mask[int(h * 0.25):int(h * 0.42), :] = 0
    mask[int(h * 0.72):int(h * 0.86), int(w * 0.25):int(w * 0.75)] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=2)
    return mask


# ----------------------------------------------------------------------------
# Scoring  (all return 0-100, higher = more visible concern)
# ----------------------------------------------------------------------------
def _scale(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1) * 100)


def pigmentation_score(face_rgb, mask) -> float:
    """Unevenness of tone: local luminance variation + share of dark spots."""
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)
    # Remove lighting gradients with a large blur, keep spot-sized variation
    low = cv2.GaussianBlur(L, (0, 0), sigmaX=max(face_rgb.shape[1] / 6, 5))
    mid = cv2.GaussianBlur(L, (0, 0), sigmaX=2.0)
    detail = (mid - low)[mask > 0]
    if detail.size < 200:
        return 0.0
    spread = float(np.std(detail))
    spot_share = float(np.mean(detail < -2.0 * spread)) if spread > 0 else 0.0
    return round(0.7 * _scale(spread, 2.0, 12.0) + 0.3 * _scale(spot_share, 0.01, 0.08), 1)


def texture_score(face_rgb, mask) -> float:
    """Micro-roughness: high-frequency energy of luminance on skin."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Normalise face size so score is resolution independent
    scale = 256.0 / face_rgb.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 0.8), cv2.CV_32F)
    vals_hp = np.abs(hp)[m > 0]
    vals_lap = np.abs(lap)[m > 0]
    if vals_hp.size < 200:
        return 0.0
    return round(0.6 * _scale(vals_hp.mean(), 0.8, 4.0) + 0.4 * _scale(vals_lap.mean(), 0.8, 5.0), 1)


def redness_score(face_rgb, mask) -> float:
    """Red dominance via LAB a* channel, relative to the person's own baseline."""
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float32)[mask > 0]
    if a.size < 200:
        return 0.0
    baseline = np.median(a)
    elevated_share = float(np.mean(a > baseline + 6))
    mean_a = float(np.mean(a)) - 128  # >0 means reddish
    return round(0.5 * _scale(mean_a, 4, 24) + 0.5 * _scale(elevated_share, 0.03, 0.30), 1)


def estimate_skin_age(chrono_age, pig, tex, red) -> int:
    """
    Heuristic: weighted blend of concerns shifts age around the stated age.
    Neutral point is ~35/100; range capped to +/- 12 years.
    """
    composite = 0.40 * pig + 0.40 * tex + 0.20 * red
    shift = (composite - 35) / 65 * 12
    return int(round(chrono_age + float(np.clip(shift, -12, 12))))


def quality_checks(rgb, face_rgb, mask) -> list:
    notes = []
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 70:
        notes.append("Image is dark – use even daylight for better accuracy.")
    if gray.mean() > 200:
        notes.append("Image is overexposed – avoid direct flash.")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        notes.append("Image looks blurry – hold the camera steady and refocus.")
    if face_rgb.shape[1] < 160:
        notes.append("Face is small in the frame – move closer.")
    if (mask > 0).mean() < 0.15:
        notes.append("Little visible skin detected – remove glasses/filters and face the camera.")
    return notes


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def analyze(rgb: np.ndarray, chrono_age: int):
    box = detect_face(rgb)
    if box is None:
        return None
    x, y, w, h = box
    pad = int(0.05 * w)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, rgb.shape[1]), min(y + h + pad, rgb.shape[0])
    face = rgb[y0:y1, x0:x1]

    mask = skin_mask(face)
    pig = pigmentation_score(face, mask)
    tex = texture_score(face, mask)
    red = redness_score(face, mask)
    age = estimate_skin_age(chrono_age, pig, tex, red)

    # Annotated image: face box + green skin-region overlay
    annotated = rgb.copy()
    overlay = annotated[y0:y1, x0:x1].copy()
    overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([60, 200, 120])).astype(np.uint8)
    annotated[y0:y1, x0:x1] = overlay
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 3)

    return Analysis(pig, tex, red, age, chrono_age, box, annotated,
                    quality_checks(rgb, face, mask))


def recommend(a: Analysis) -> list:
    """Return [(product, reasons)] ordered by relevance."""
    concerns = {
        "pigmentation": a.pigmentation,
        "texture": a.texture,
        "redness": a.redness,
        "age": max(0, (a.skin_age - a.chronological_age)) / 12 * 100,
    }
    nice = {"pigmentation": "uneven tone", "texture": "skin texture",
            "redness": "redness", "age": "skin age above your actual age"}
    ranked = []
    for name, info in PRODUCTS.items():
        hits = [t for t in info["targets"] if concerns[t] >= CONCERN_THRESHOLD]
        if hits:
            ranked.append((sum(concerns[t] for t in hits), name,
                           [nice[t] for t in hits]))
    ranked.sort(reverse=True)
    return [(n, r) for _, n, r in ranked]


def consultation_text(a: Analysis, recs: list) -> dict:
    """Build the written report sections (shared by UI and PDF)."""
    scores = {"Pigmentation": a.pigmentation, "Texture": a.texture, "Redness": a.redness}
    top = max(scores, key=scores.get)
    delta = a.skin_age - a.chronological_age

    if delta > 2:
        age_line = f"Estimated skin age is {a.skin_age}, about {delta} years above your stated age of {a.chronological_age}."
    elif delta < -2:
        age_line = f"Estimated skin age is {a.skin_age}, about {abs(delta)} years below your stated age of {a.chronological_age}."
    else:
        age_line = f"Estimated skin age is {a.skin_age}, in line with your stated age of {a.chronological_age}."

    if max(scores.values()) < CONCERN_THRESHOLD:
        summary = "Overall the analysis shows few visible concerns. A maintenance routine is appropriate."
    else:
        summary = f"The most prominent area is {top.lower()} ({label(scores[top]).lower()}, {scores[top]:.0f}/100)."

    findings = [
        f"Pigmentation: {label(a.pigmentation)} ({a.pigmentation:.0f}/100) – variation in tone and dark-spot coverage.",
        f"Texture: {label(a.texture)} ({a.texture:.0f}/100) – surface roughness and fine-detail irregularity.",
        f"Redness: {label(a.redness)} ({a.redness:.0f}/100) – red tone relative to your own skin baseline.",
    ]

    if recs:
        rec_lines = [f"{n} – {PRODUCTS[n]['role']} Addresses: {', '.join(r)}." for n, r in recs]
    else:
        rec_lines = ["No product is flagged at this time. Consider a gentle maintenance routine and daily SPF."]

    tips = [
        "Use broad-spectrum SPF daily – sun exposure drives pigmentation and visible ageing.",
        "Introduce one new product at a time and patch-test before full use.",
        "Re-analyse under the same lighting every 4–6 weeks to track change.",
    ]
    return {"age_line": age_line, "summary": summary, "findings": findings,
            "recs": rec_lines, "tips": tips}


# ----------------------------------------------------------------------------
# Charts (matplotlib -> PNG bytes, reused in the app and the PDF)
# ----------------------------------------------------------------------------
INK, TEAL, MUTED, GRID = "#14272E", "#1B6B73", "#5C6F75", "#D9E2E4"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _radar(pig, tex, red, ageing):
    labels = ["Pigmentation", "Texture", "Redness", "Ageing"]
    vals = [pig, tex, red, ageing]
    ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
    ang_c, vals_c = ang + ang[:1], vals + vals[:1]
    fig, ax = plt.subplots(figsize=(4.4, 4.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color=MUTED)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.grid(color=GRID)
    ax.spines["polar"].set_color(GRID)
    ax.plot(ang_c, [CONCERN_THRESHOLD] * len(ang_c), color="#B58A1B", ls="--", lw=1.2,
            label=f"Concern line ({CONCERN_THRESHOLD})")
    ax.plot(ang_c, vals_c, color=TEAL, lw=2.2, label="Your scores")
    ax.fill(ang_c, vals_c, color=TEAL, alpha=0.2)
    for a_, v in zip(ang, vals):
        ax.text(a_, min(v + 9, 108), f"{v:.0f}", ha="center", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_title("Skin profile", fontsize=12, color=INK, pad=18, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8)
    return _png(fig)


def _bars(pig, tex, red):
    names = ["Redness", "Texture", "Pigmentation"]   # bottom to top
    vals = [red, tex, pig]
    cols = [LEVEL_COLORS[label(v)] for v in vals]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.barh(names, vals, color=cols, height=0.55)
    ax.axvline(CONCERN_THRESHOLD, color="#B58A1B", ls="--", lw=1.2)
    ax.text(CONCERN_THRESHOLD, 2.52, "concern line", fontsize=7.5, color="#B58A1B", ha="center", va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.9)
    for i, v in enumerate(vals):
        ax.text(min(v + 2, 80), i, f"{v:.0f}  {label(v)}", va="center", fontsize=9, color=INK)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Scores (higher = more visible concern)", fontsize=12, color=INK,
                 loc="left", fontweight="bold", pad=12)
    return _png(fig)


def _age(skin_age, chrono):
    delta = skin_age - chrono
    col = "#B23A48" if delta > 2 else ("#3E8E6B" if delta < -2 else TEAL)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    bars = ax.bar(["Stated age", "Skin age\nestimate"], [chrono, skin_age], color=["#AFC3C7", col], width=0.55)
    for b, v in zip(bars, [chrono, skin_age]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}", ha="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.set_ylim(0, max(chrono, skin_age) * 1.3)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="x", labelcolor=INK)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"Skin age {delta:+d} yrs vs stated", fontsize=12, color=INK, fontweight="bold", pad=12)
    return _png(fig)


@lru_cache(maxsize=16)
def _charts_cached(pig, tex, red, skin_age, chrono):
    ageing = max(0, skin_age - chrono) / 12 * 100
    return {"radar": _radar(pig, tex, red, ageing), "bars": _bars(pig, tex, red),
            "age": _age(skin_age, chrono)}


def score_charts(a: Analysis) -> dict:
    """PNG bytes for the radar, bar and skin-age charts."""
    return _charts_cached(round(a.pigmentation, 1), round(a.texture, 1), round(a.redness, 1),
                          int(a.skin_age), int(a.chronological_age))


CHANGE_MIN = 3.0   # score changes smaller than this are treated as "no clear change"


def _compare_scores(b, a):
    names = ["Pigmentation", "Texture", "Redness"]
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    b1 = ax.bar(x - w / 2, b, w, color="#AFC3C7", label="1st photo")
    b2 = ax.bar(x + w / 2, a, w, color=TEAL, label="2nd photo")
    ax.axhline(CONCERN_THRESHOLD, color="#B58A1B", ls="--", lw=1.2)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                    ha="center", fontsize=9, color=INK)
    for i, (vb, va) in enumerate(zip(b, a)):
        d = va - vb
        if abs(d) < CHANGE_MIN:
            txt, col = "no clear\nchange", MUTED
        else:
            txt, col = f"{d:+.0f}", LEVEL_COLORS["Low"] if d < 0 else LEVEL_COLORS["High"]
        ax.text(i, max(vb, va) + 10, txt, ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, color=INK)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.set_title("Scores: 1st photo vs 2nd photo (lower = less visible concern)", fontsize=11,
                 color=INK, loc="left", fontweight="bold", pad=12)
    return _png(fig)


def _compare_age(chrono, age_b, age_a):
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    vals = [chrono, age_b, age_a]
    bars = ax.bar(["Stated\nage", "1st\nphoto", "2nd\nphoto"], vals,
                  color=["#DCE6E8", "#AFC3C7", TEAL], width=0.55)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v}", ha="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="x", labelcolor=INK)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"Skin age estimate: {age_a - age_b:+d} yrs", fontsize=11, color=INK,
                 fontweight="bold", pad=12)
    return _png(fig)


@lru_cache(maxsize=16)
def _compare_cached(b, a, chrono, age_b, age_a):
    return {"scores": _compare_scores(b, a), "age": _compare_age(chrono, age_b, age_a)}


def compare_score_charts(rb: Analysis, ra: Analysis) -> dict:
    return _compare_cached((round(rb.pigmentation, 1), round(rb.texture, 1), round(rb.redness, 1)),
                           (round(ra.pigmentation, 1), round(ra.texture, 1), round(ra.redness, 1)),
                           int(rb.chronological_age), int(rb.skin_age), int(ra.skin_age))


def change_rows(rb: Analysis, ra: Analysis) -> list:
    """[(name, before, after, delta, status)] where status is lower / higher / same."""
    rows = []
    for name, vb, va in (("Pigmentation", rb.pigmentation, ra.pigmentation),
                         ("Texture", rb.texture, ra.texture),
                         ("Redness", rb.redness, ra.redness)):
        d = va - vb
        rows.append((name, vb, va, d, "same" if abs(d) < CHANGE_MIN else ("lower" if d < 0 else "higher")))
    return rows


def change_lines(rb: Analysis, ra: Analysis) -> list:
    out = []
    for name, vb, va, d, st_ in change_rows(rb, ra):
        if st_ == "same":
            out.append(f"{name}: {vb:.0f} to {va:.0f}, no clear change.")
        else:
            out.append(f"{name}: {vb:.0f} to {va:.0f}, {abs(d):.0f} points {st_} (visible concern).")
    da = ra.skin_age - rb.skin_age
    out.append("Skin age estimate: " + ("no change." if da == 0 else
                                        f"{abs(da)} year{'s' if abs(da) != 1 else ''} "
                                        f"{'younger' if da < 0 else 'older'} than in the 1st photo."))
    return out


# ----------------------------------------------------------------------------
# Face contour comparison (real detection, no simulation)
# MediaPipe Face Landmarker (free, Apache-2.0) finds 468 face landmarks. The 36
# "face oval" landmarks are the contour. Photo 1 is drawn in RED, photo 2 in
# GREEN after being aligned to photo 1 using rigid points (eye corners and nose
# bridge), so differences along the jawline are real differences between photos.
# ----------------------------------------------------------------------------
RED = (235, 64, 52)
GREEN = (46, 204, 113)

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAW = [323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
ALIGN = [33, 133, 362, 263, 168, 6]   # eye corners + nose bridge: stay put when the jaw changes
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

_cache_resource = getattr(st, "cache_resource", lambda f: f)
_cache_data = getattr(st, "cache_data", lambda f: f)
_LOCK = threading.Lock()


class LandmarkUnavailable(RuntimeError):
    """Face landmark model could not be loaded on this server."""


def _model_path() -> str:
    # A copy placed next to app.py (same folder in the repository) is used first
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
    if os.path.exists(local) and os.path.getsize(local) > 1_000_000:
        return local
    path = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        tmp = path + ".part"
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, path)
    return path


@_cache_resource
def _get_landmarker():
    import mediapipe as mp
    if hasattr(mp, "solutions"):   # older releases ship the model inside the package
        return "legacy", mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
        num_faces=1, running_mode=vision.RunningMode.IMAGE)
    return "tasks", vision.FaceLandmarker.create_from_options(opts)


@_cache_data
def face_landmarks(rgb):
    """(468, 2) float32 pixel landmarks of the main face, or None if no face."""
    try:
        kind, lm = _get_landmarker()
    except Exception as e:                      # missing library / no model / no network
        raise LandmarkUnavailable(f"{type(e).__name__}: {e}")
    h, w = rgb.shape[:2]
    with _LOCK:
        if kind == "legacy":
            res = lm.process(np.ascontiguousarray(rgb))
            pts = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None
        else:
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = lm.detect(img)
            pts = res.face_landmarks[0] if res.face_landmarks else None
    if pts is None:
        return None
    return np.array([[p.x * w, p.y * h] for p in pts[:468]], dtype=np.float32)


def yaw_ratio(lm) -> float:
    """~0.5 when the face looks straight at the camera."""
    return float((lm[1, 0] - lm[234, 0]) / max(lm[454, 0] - lm[234, 0], 1e-6))


def compare_faces(lm_b, lm_a):
    """Align photo 2 to photo 1 and measure the jawline difference."""
    M, _ = cv2.estimateAffinePartial2D(lm_a[ALIGN], lm_b[ALIGN], method=cv2.LMEDS)
    if M is None:
        return None
    aligned = lm_a @ M[:, :2].T + M[:, 2]
    fh = float(np.linalg.norm(lm_b[10] - lm_b[152]))
    d = aligned[JAW] - lm_b[JAW]
    jaw_w_b = float(np.linalg.norm(lm_b[172] - lm_b[397]))
    jaw_w_a = float(np.linalg.norm(aligned[172] - aligned[397]))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    yaw_diff = abs(yaw_ratio(lm_a) - yaw_ratio(lm_b))
    notes = []
    if not 0.85 <= scale <= 1.18:
        notes.append("The photos were taken at different distances. Lines were scaled to match, "
                     "but camera distance can change the apparent contour.")
    if abs(angle) > 8:
        notes.append("The head is tilted differently in the two photos.")
    if yaw_diff > 0.05:
        notes.append("The head is turned differently in the two photos, which can change the contour by itself.")
    return {
        "M": M, "aligned": aligned.astype(np.float32),
        "lift_pct": float(-np.mean(d[:, 1]) / fh * 100),          # + means higher in photo 2
        "max_pct": float(np.max(np.linalg.norm(d, axis=1)) / fh * 100),
        "width_pct": float((jaw_w_a - jaw_w_b) / max(jaw_w_b, 1e-6) * 100),
        "scale": scale, "angle": angle, "yaw_diff": yaw_diff, "notes": notes,
    }


def metrics_lines(m) -> list:
    lift = m["lift_pct"]
    if abs(lift) < 1.0:
        first = "Jawline height: no meaningful change (within detection error)."
    elif lift > 0:
        first = f"Jawline sits about {lift:.1f}% of face height higher in the 2nd photo."
    else:
        first = f"Jawline sits about {abs(lift):.1f}% of face height lower in the 2nd photo."
    return [first,
            f"Largest local difference along the jawline: {m['max_pct']:.1f}% of face height.",
            f"Jaw width: {m['width_pct']:+.1f}% compared with the 1st photo."]


def smooth_closed(pts, n=8):
    """Catmull-Rom curve through the points (closed loop) so lines look smooth."""
    P = np.asarray(pts, np.float32)
    m = len(P)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out, np.float32)


def draw_grid(img, box, cells):
    """Grid spans the face box: top line = forehead, bottom line = chin."""
    x, y, w, h = box
    H, W = img.shape[:2]
    gx0, gx1 = max(x, 0), min(x + w, W - 1)
    gy0, gy1 = max(y, 0), min(y + h, H - 1)
    th = max(1, int(round(w / 260)))
    layer = img.copy()
    for xi in np.linspace(gx0, gx1, cells + 1):
        cv2.line(layer, (int(xi), int(gy0)), (int(xi), int(gy1)), (255, 255, 255), th, cv2.LINE_AA)
    for yi in np.linspace(gy0, gy1, cells + 1):
        cv2.line(layer, (int(gx0), int(yi)), (int(gx1), int(yi)), (255, 255, 255), th, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.55, img, 0.45, 0)


def draw_contour(img, pts, color, th):
    p = smooth_closed(pts).astype(np.int32)
    cv2.polylines(img, [p.reshape(-1, 1, 2)], True, color, th, cv2.LINE_AA)


def contour_views(rgb_b, lm_b, rgb_a=None, lm_a=None, show_grid=True, cells=12, show_diff=True):
    """
    Returns dict with 'before' (red contour), and when a 2nd photo is given:
    'after' (green contour, aligned to photo 1), 'overlay' (both + shaded difference), 'metrics'.
    All images are cropped around the face of photo 1.
    """
    H, W = rgb_b.shape[:2]
    x0, y0 = lm_b[OVAL].min(axis=0)
    x1, y1 = lm_b[OVAL].max(axis=0)
    box = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    th = max(2, int(round(box[2] / 170)))

    base = draw_grid(rgb_b, box, cells) if show_grid else rgb_b.copy()
    before = base.copy()
    draw_contour(before, lm_b[OVAL], RED, th)

    mx, my = int(0.3 * box[2]), int(0.25 * box[3])
    cx0, cx1 = max(box[0] - mx, 0), min(box[0] + box[2] + mx, W)
    cy0, cy1 = max(box[1] - my, 0), min(box[1] + box[3] + my, H)
    crop = lambda im: im[cy0:cy1, cx0:cx1]

    out = {"before": crop(before), "metrics": None, "after": None, "overlay": None}
    if rgb_a is None or lm_a is None:
        return out
    cmp_ = compare_faces(lm_b, lm_a)
    if cmp_ is None:
        return out
    aligned_img = cv2.warpAffine(rgb_a, cmp_["M"], (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    after = draw_grid(aligned_img, box, cells) if show_grid else aligned_img.copy()
    draw_contour(after, cmp_["aligned"][OVAL], GREEN, th)

    overlay = rgb_b.copy()
    if show_diff:
        shade = np.zeros((H, W), np.uint8)
        band = np.vstack([lm_b[JAW], cmp_["aligned"][JAW][::-1]]).astype(np.int32)
        cv2.fillPoly(shade, [band.reshape(-1, 1, 2)], 255)
        shade = cv2.GaussianBlur(shade, (0, 0), max(2, box[2] / 70))
        alpha = (shade.astype(np.float32) / 255 * 0.6)[..., None]
        overlay = (overlay * (1 - alpha) + np.array([20, 30, 60], np.float32) * alpha).astype(np.uint8)
    draw_contour(overlay, lm_b[OVAL], RED, th)
    draw_contour(overlay, cmp_["aligned"][OVAL], GREEN, th)

    out.update(after=crop(after), overlay=crop(overlay), metrics=cmp_)
    return out


# ----------------------------------------------------------------------------
# PDF export
# ----------------------------------------------------------------------------
def _png_image(data: bytes, width):
    im = Image.open(io.BytesIO(data))
    return RLImage(io.BytesIO(data), width=width, height=width * im.height / im.width)


def _rl_image(arr, width):
    img = Image.fromarray(arr)
    img.thumbnail((900, 900))
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=88)
    b.seek(0)
    return RLImage(b, width=width, height=width * img.height / img.width)


def build_pdf(a: Analysis, recs: list, client_name: str, lift: dict = None,
              after: Analysis = None) -> bytes:
    text = consultation_text(a, recs)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Skin Consultation Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#14272E"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#1B6B73"), spaceBefore=12)
    body = ParagraphStyle("b", parent=styles["BodyText"], leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)

    band = Table([[Paragraph("Skin Consultation Report",
                         ParagraphStyle("hb", parent=h1, textColor=colors.white, alignment=0,
                                        fontSize=20, leading=24))]], colWidths=[174 * mm])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#14272E")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, -1), 12),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story = [band, Spacer(1, 6)]
    meta = f"Client: {client_name or 'N/A'} &nbsp;|&nbsp; Date: {datetime.now():%d %b %Y}"
    story += [Paragraph(meta, body), Spacer(1, 8)]

    # Annotated image
    img = Image.fromarray(a.annotated)
    img.thumbnail((700, 700))
    ib = io.BytesIO()
    img.save(ib, format="JPEG", quality=88)
    ib.seek(0)
    ratio = img.height / img.width
    iw = 70 * mm
    story += [RLImage(ib, width=iw, height=iw * ratio), Spacer(1, 8)]

    # Score table
    rows = [["Metric", "Score", "Level"],
            ["Pigmentation", f"{a.pigmentation:.0f}/100", label(a.pigmentation)],
            ["Texture", f"{a.texture:.0f}/100", label(a.texture)],
            ["Redness", f"{a.redness:.0f}/100", label(a.redness)],
            ["Skin age estimate", f"{a.skin_age} yrs", f"Stated age {a.chronological_age}"]]
    t = Table(rows, colWidths=[60 * mm, 40 * mm, 60 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14272E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2E4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F7")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ] + [("TEXTCOLOR", (2, i), (2, i), colors.HexColor(LEVEL_COLORS[label(v)]))
         for i, v in ((1, a.pigmentation), (2, a.texture), (3, a.redness))]))
    story += [t, Paragraph("Summary", h2), Paragraph(text["summary"], body),
              Paragraph(text["age_line"], body), Paragraph("Findings", h2)]
    story += [Paragraph("• " + f, body) for f in text["findings"]]
    ch = score_charts(a)
    ct = Table([[_png_image(ch["radar"], 78 * mm), _png_image(ch["bars"], 92 * mm)],
                [_png_image(ch["age"], 58 * mm), ""]], colWidths=[84 * mm, 90 * mm])
    ct.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story += [KeepTogether([Paragraph("Charts", h2), ct])]
    if after is not None:
        cc = compare_score_charts(a, after)
        cmp_t = Table([[_png_image(cc["scores"], 98 * mm), _png_image(cc["age"], 62 * mm)]],
                      colWidths=[102 * mm, 70 * mm])
        cmp_t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story += [KeepTogether([Paragraph("Before and after skin scores", h2), cmp_t]),
                  Spacer(1, 4)]
        story += [Paragraph("• " + t_, body) for t_ in change_lines(a, after)]
        story += [Spacer(1, 3), Paragraph(
            "Scores are affected by lighting and camera settings. Compare only photos taken under the "
            "same conditions. Changes under 3 points are treated as no clear change.", small)]
    story += [Paragraph("Recommended products", h2)]
    story += [Paragraph("• " + r, body) for r in text["recs"]]
    story += [Paragraph("Care guidance", h2)]
    story += [Paragraph("• " + tip, body) for tip in text["tips"]]
    if lift:
        story += [PageBreak(), Paragraph("Face contour comparison", h2)]
        if lift.get("after") is not None:
            pair = Table([[_rl_image(lift["before"], 82 * mm), _rl_image(lift["after"], 82 * mm)],
                          ["1st photo: contour in red", "2nd photo (aligned): contour in green"]],
                         colWidths=[86 * mm, 86 * mm])
            pair.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                      ("FONTSIZE", (0, 1), (-1, 1), 9)]))
            story += [pair, Spacer(1, 8), _rl_image(lift["overlay"], 90 * mm),
                      Paragraph("Both contours on the 1st photo. Red = 1st photo, green = 2nd photo, "
                                "shaded band = difference along the jawline.", small), Spacer(1, 6)]
            story += [Paragraph("• " + t_, body) for t_ in lift["lines"]]
            story += [Paragraph("• " + n_, body) for n_ in lift["notes"]]
        else:
            story += [_rl_image(lift["before"], 90 * mm),
                      Paragraph("Face contour detected in the 1st photo (red).", small)]
        story += [Spacer(1, 6), Paragraph(
            "Contours are detected automatically and are approximate. Differences between photos can "
            "come from head angle, expression, lighting and camera distance, not only from treatment. "
            "This is not a medical measurement or a guarantee of any result.", small)]
    story += [Spacer(1, 14), Paragraph(
        "This report is generated from an automated image analysis for cosmetic consultation only. "
        "It is not a medical diagnosis. Consult a qualified professional for skin conditions or concerns.",
        small)]
    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
:root{--ink:#14272E;--teal:#1B6B73;--teal-dk:#14545B;--mist:#F4F7F7;--line:#D9E2E4;--muted:#5C6F75;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--mist);color:var(--ink);font-family:'Figtree',system-ui,sans-serif;}
.block-container{max-width:1180px;padding-top:1.4rem;padding-bottom:3rem;}
footer{visibility:hidden;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
[class*="st-key-card"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.2rem 1.3rem;margin-bottom:1.1rem;}
[data-testid="stImage"] img{border-radius:12px;}
.hero{background:var(--ink);border-radius:16px;padding:1.7rem 1.9rem;margin-bottom:1.1rem;}
.hero-title{font-family:'Fraunces',Georgia,serif;font-size:2.1rem;line-height:1.15;color:#fff;font-weight:600;}
.hero-sub{color:#B9CDD1;max-width:62ch;margin-top:.45rem;font-size:1rem;line-height:1.5;}
.sec-title{font-family:'Fraunces',Georgia,serif;font-size:1.4rem;font-weight:600;color:var(--ink);margin:0 0 .15rem;}
.sec-sub{color:var(--muted);font-size:.93rem;margin-bottom:1rem;max-width:70ch;}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;}
.tile{background:#F7FAFA;border:1px solid var(--line);border-radius:12px;padding:.95rem 1.05rem;}
.tile-label{font-size:.86rem;color:var(--muted);font-weight:500;}
.tile-value{font-family:'Fraunces',Georgia,serif;font-size:2.2rem;line-height:1.1;margin:.2rem 0 .55rem;color:var(--ink);}
.tile-value span{font-family:'Figtree',sans-serif;font-size:.9rem;color:var(--muted);margin-left:.25rem;}
.bar{height:6px;background:#E3EBEC;border-radius:99px;overflow:hidden;}
.bar i{display:block;height:100%;border-radius:99px;}
.tile-level{margin-top:.55rem;font-size:.86rem;font-weight:600;}
.callout{background:#EAF3F4;border-radius:12px;padding:1rem 1.1rem;line-height:1.55;margin-bottom:1rem;}
.find{list-style:none;padding:0;margin:0 0 1rem;}
.find li{display:flex;gap:.7rem;align-items:flex-start;padding:.6rem 0;border-bottom:1px solid var(--line);line-height:1.45;}
.find li:last-child{border-bottom:0;}
.dot{flex:0 0 10px;height:10px;border-radius:50%;margin-top:.45rem;}
.prod{border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:12px;padding:.95rem 1.05rem;background:#fff;height:100%;}
.prod-name{font-weight:600;font-size:1rem;}
.prod-role{color:var(--muted);font-size:.9rem;margin:.3rem 0 .6rem;line-height:1.45;}
.tag{display:inline-block;background:#E6F1F2;color:var(--teal-dk);border-radius:99px;padding:.15rem .65rem;font-size:.78rem;margin:0 .3rem .25rem 0;font-weight:500;}
.chip{display:inline-block;background:#F7FAFA;border:1px solid var(--line);border-radius:99px;padding:.35rem .9rem;font-size:.88rem;margin:.4rem 0;}
.subhead{font-weight:600;margin:1.1rem 0 .5rem;}
button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--teal)!important;border:0!important;color:#fff!important;border-radius:10px!important;font-weight:600!important;padding:.65rem 1.3rem!important;}
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:var(--teal-dk)!important;}
button[kind="secondary"],button[data-testid="stBaseButton-secondary"]{border-radius:10px!important;border:1px solid var(--line)!important;}
[data-testid="stSlider"] [role="slider"]{background-color:var(--teal)!important;}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;background:#FBFDFD;}
@media (max-width:640px){.hero{padding:1.3rem 1.2rem;}.hero-title{font-size:1.7rem;}[class*="st-key-card"]{padding:1rem;}}
</style>"""

FINDING_TEXT = {
    "Pigmentation": "Variation in tone and share of dark spots.",
    "Texture": "Surface roughness and fine-detail irregularity.",
    "Redness": "Red tone relative to your own skin baseline.",
}


PRESENT_CSS = """<style>
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none!important;}
.st-key-card_photo,.st-key-ops_lift,.st-key-ops_notes{display:none!important;}
</style>"""


def hero_html(name=""):
    who = f" Prepared for {html.escape(name.strip())}." if name and name.strip() else ""
    return ('<div class="hero"><div class="hero-title">Skin consultation</div>'
            '<div class="hero-sub">Review pigmentation, texture and redness from a selfie, preview a lifting '
            f'result, and export a report.{who}</div></div>')


def section(title, sub=""):
    st.markdown(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>',
                unsafe_allow_html=True)


def tile_html(name, value, unit="/100", level=None, sub=None, color=None, pct=None):
    level_txt = level if level is not None else sub
    color = color or LEVEL_COLORS.get(level, "#1B6B73")
    pct = value if pct is None else pct
    return (f'<div class="tile"><div class="tile-label">{name}</div>'
            f'<div class="tile-value">{value:.0f}<span>{unit}</span></div>'
            f'<div class="bar"><i style="width:{max(0, min(100, pct)):.0f}%;background:{color}"></i></div>'
            f'<div class="tile-level" style="color:{color}">{level_txt}</div></div>')


def tiles_block(r: Analysis) -> str:
    delta = r.skin_age - r.chronological_age
    age_color = "#B23A48" if delta > 2 else ("#3E8E6B" if delta < -2 else "#1B6B73")
    age_sub = f"{delta:+d} yrs vs stated age {r.chronological_age}"
    age_pct = (r.skin_age - 10) / 80 * 100
    return ('<div class="tiles">'
            + tile_html("Pigmentation", r.pigmentation, level=label(r.pigmentation))
            + tile_html("Texture", r.texture, level=label(r.texture))
            + tile_html("Redness", r.redness, level=label(r.redness))
            + tile_html("Skin age estimate", r.skin_age, unit="yrs", sub=age_sub, color=age_color, pct=age_pct)
            + '</div>')


def findings_block(r: Analysis) -> str:
    rows = ""
    for name, v in (("Pigmentation", r.pigmentation), ("Texture", r.texture), ("Redness", r.redness)):
        lv = label(v)
        c = LEVEL_COLORS[lv]
        rows += (f'<li><span class="dot" style="background:{c}"></span><div><b>{name}</b> '
                 f'<span style="color:{c};font-weight:600">{lv}, {v:.0f}/100</span><br>'
                 f'<span style="color:#5C6F75">{FINDING_TEXT[name]}</span></div></li>')
    return f'<ul class="find">{rows}</ul>'


def product_block(name, reasons) -> str:
    tags = "".join(f'<span class="tag">{r}</span>' for r in reasons)
    return (f'<div class="prod"><div class="prod-name">{name}</div>'
            f'<div class="prod-role">{PRODUCTS[name]["role"]}</div>{tags}</div>')


def change_tiles(rb: Analysis, ra: Analysis) -> str:
    """Tiles showing the change between the 1st and 2nd photo."""
    def tile(name, text, unit, level, color):
        return (f'<div class="tile"><div class="tile-label">{name}</div>'
                f'<div class="tile-value">{text}<span>{unit}</span></div>'
                f'<div class="tile-level" style="color:{color}">{level}</div></div>')
    palette = {"lower": ("Lower concern", LEVEL_COLORS["Low"]),
               "higher": ("Higher concern", LEVEL_COLORS["High"]),
               "same": ("No clear change", "#5C6F75")}
    html_ = '<div class="tiles">'
    for name, vb, va, d, st_ in change_rows(rb, ra):
        lvl, col = palette[st_]
        html_ += tile(name, f"{d:+.0f}", " points", lvl, col)
    da = ra.skin_age - rb.skin_age
    col = LEVEL_COLORS["Low"] if da < 0 else (LEVEL_COLORS["High"] if da > 0 else "#5C6F75")
    html_ += tile("Skin age estimate", f"{da:+d}", " yrs",
                  "Younger" if da < 0 else ("Older" if da > 0 else "No change"), col)
    return html_ + "</div>"


def main():
    st.set_page_config(page_title="Skin consultation", page_icon="🧴", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    n_reset = st.session_state.get("reset_n", 0)

    with st.sidebar:
        st.markdown('<div class="sec-title">Client details</div>', unsafe_allow_html=True)
        name = st.text_input("Name (optional)", key="client_name")
        age = st.number_input("Age", min_value=10, max_value=90, value=30, step=1, key="client_age")
        st.markdown('<div class="subhead">For the best photo</div>', unsafe_allow_html=True)
        st.markdown("- Natural, even light\n- No filters or beauty mode\n"
                    "- Face the camera, hair off the forehead\n- Remove glasses")

    st.markdown(hero_html(name), unsafe_allow_html=True)
    bar1, bar2, bar3 = st.columns([2.2, 1, 1.1])
    bar1.caption("Set the photo and grid first, then switch on Presentation mode before showing the client.")
    present = bar2.toggle("Presentation mode", key="present")
    if bar3.button("Reset for next client", use_container_width=True):
        st.session_state.clear()
        st.session_state["reset_n"] = n_reset + 1
        st.rerun()

    # ---- 1. Photos --------------------------------------------------------
    with st.container(key="card_photo"):
        section("Photos", "Add the 1st photo for the skin analysis. Add a 2nd photo, for example after "
                          "treatment, to compare the face contour.")
        pc1, pc2 = st.columns(2, gap="large")
        with pc1:
            st.markdown('<div class="subhead">1st photo</div>', unsafe_allow_html=True)
            source = st.radio("Photo source", ["Upload", "Camera"], horizontal=True,
                              label_visibility="collapsed")
            file = (st.file_uploader("Upload 1st photo", type=["jpg", "jpeg", "png", "webp"],
                                     label_visibility="collapsed", key=f"upload_{n_reset}")
                    if source == "Upload"
                    else st.camera_input("Take 1st photo", label_visibility="collapsed",
                                         key=f"camera_{n_reset}"))
        with pc2:
            st.markdown('<div class="subhead">2nd photo (optional)</div>', unsafe_allow_html=True)
            file2 = st.file_uploader("Upload 2nd photo", type=["jpg", "jpeg", "png", "webp"],
                                     label_visibility="collapsed", key=f"upload2_{n_reset}")
            st.caption("Use the same angle, distance and light as the 1st photo.")

    if not file:
        st.info("Add the 1st photo above to start the analysis.")
        st.caption(f"Version: {APP_VERSION}")
        return

    if present:
        st.markdown(PRESENT_CSS, unsafe_allow_html=True)

    rgb = load_image(file)
    try:
        with st.spinner("Analysing skin…"):
            result = analyze(rgb, int(age))
    except RuntimeError as e:
        st.error(str(e))
        return

    if result is None:
        st.image(rgb, caption="Uploaded image", width=360)
        st.error("No face detected. Use a clear, front-facing photo with good lighting.")
        return

    recs = recommend(result)
    text = consultation_text(result, recs)

    rgb2, result2 = None, None
    if file2 is not None:
        rgb2 = load_image(file2)
        try:
            result2 = analyze(rgb2, int(age))
        except RuntimeError:
            result2 = None

    # ---- 2. Skin analysis -------------------------------------------------
    with st.container(key="card_analysis"):
        section("Skin analysis", "Higher scores mean more visible concern. Skin age is an estimate.")
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            st.image(result.annotated, caption="Detected face and analysed skin region (green)",
                     use_container_width=True)
        with right:
            st.markdown(tiles_block(result), unsafe_allow_html=True)
            with st.container(key="ops_notes"):
                for n in result.quality_notes:
                    st.warning(n)

    # ---- Charts ------------------------------------------------------------
    with st.container(key="card_charts"):
        section("Score charts", "The same results as charts. The dashed line marks where a score becomes a concern.")
        ch = score_charts(result)
        c1, c2, c3 = st.columns([1, 1.25, 0.85], gap="medium")
        c1.image(ch["radar"], use_container_width=True)
        c2.image(ch["bars"], use_container_width=True)
        c3.image(ch["age"], use_container_width=True)

    # ---- Before / after skin scores ---------------------------------------
    if result2 is not None:
        with st.container(key="card_compare"):
            section("Before and after skin scores",
                    "The same analysis run on the 2nd photo. Lower scores mean less visible concern.")
            cc = compare_score_charts(result, result2)
            cx1, cx2 = st.columns([1.5, 1], gap="medium")
            cx1.image(cc["scores"], use_container_width=True)
            cx2.image(cc["age"], use_container_width=True)
            st.markdown(change_tiles(result, result2), unsafe_allow_html=True)
            st.caption("Scores depend on lighting and camera settings, so compare only photos taken under the "
                       "same conditions. Changes under 3 points are treated as no clear change.")
    elif file2 is not None:
        st.warning("The skin scores could not be calculated for the 2nd photo because no face was detected "
                   "in it. Use a clear, front-facing photo.")

    # ---- 3. Face contour comparison --------------------------------------
    lift = None
    with st.container(key="card_lift"):
        section("Face contour comparison",
                "Red line: face contour detected in the 1st photo. Green line: contour in the 2nd photo, "
                "aligned to the 1st. Shaded area: the difference along the jawline.")
        with st.container(key="ops_lift"):
            k1, k2, k3 = st.columns(3)
            show_grid = k1.checkbox("Grid lines", value=True)
            cells = k2.slider("Grid density", 6, 20, 12)
            show_diff = k3.checkbox("Shade the difference", value=True)

        try:
            lm1 = face_landmarks(rgb)
            lm2 = face_landmarks(rgb2) if rgb2 is not None else None
        except LandmarkUnavailable as e:
            lm1 = lm2 = None
            st.warning("Face contour detection is not available on this server right now, so this section "
                       "is skipped. The skin analysis and report still work.")
            with st.expander("Technical details (send this to your developer)"):
                st.code(str(e)[:700] or "unknown error")
        else:
            if lm1 is None:
                st.warning("No face contour could be detected in the 1st photo. Use a clear, front-facing photo.")
            else:
                if abs(yaw_ratio(lm1) - 0.5) > 0.1:
                    st.warning("The face in the 1st photo is turned to one side. A straight, front-facing "
                               "photo gives a more reliable contour.")
                if rgb2 is not None and lm2 is None:
                    st.warning("No face contour could be detected in the 2nd photo. Showing the 1st photo only.")
                views = contour_views(rgb, lm1, rgb2, lm2, show_grid, cells, show_diff)
                m = views["metrics"]
                if m is None:
                    st.image(views["before"], caption="1st photo: detected face contour (red)", width=380)
                    if rgb2 is None:
                        st.info("Add a 2nd photo above to compare the contours.")
                    lift = {"before": views["before"], "after": None}
                else:
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.image(views["before"], caption="1st photo: contour in red", use_container_width=True)
                    ic2.image(views["after"], caption="2nd photo (aligned): contour in green",
                              use_container_width=True)
                    ic3.image(views["overlay"], caption="Both contours, difference shaded",
                              use_container_width=True)
                    lines = metrics_lines(m)
                    st.markdown("".join(f'<span class="chip">{html.escape(t_)}</span> ' for t_ in lines),
                                unsafe_allow_html=True)
                    for n_ in m["notes"]:
                        st.warning(n_)
                    lift = {"before": views["before"], "after": views["after"],
                            "overlay": views["overlay"], "lines": lines, "notes": m["notes"]}
        st.caption("Contours are detected automatically and are approximate. Differences can come from head "
                   "angle, expression, lighting and camera distance, not only from treatment. "
                   "This is not a medical measurement.")

    # ---- 4. Consultation report ------------------------------------------
    with st.container(key="card_report"):
        section("Consultation report", "Summary, recommended products and care guidance for this client.")
        st.markdown(f'<div class="callout">{text["summary"]}<br>{text["age_line"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="subhead">Findings</div>' + findings_block(result), unsafe_allow_html=True)

        st.markdown('<div class="subhead">Recommended products</div>', unsafe_allow_html=True)
        if recs:
            cols = st.columns(len(recs))
            for col, (pname, reasons) in zip(cols, recs):
                col.markdown(product_block(pname, reasons), unsafe_allow_html=True)
        else:
            st.write(text["recs"][0])

        st.markdown('<div class="subhead">Care guidance</div>', unsafe_allow_html=True)
        for tip in text["tips"]:
            st.markdown(f"- {tip}")

        st.write("")
        pdf = build_pdf(result, recs, name, lift, result2)
        st.download_button("Download PDF report", data=pdf,
                           file_name=f"skin_report_{datetime.now():%Y%m%d}.pdf",
                           mime="application/pdf", type="primary")

    st.caption(f"Cosmetic analysis only. Not a medical diagnosis. Version: {APP_VERSION}")


if __name__ == "__main__":
    main()
