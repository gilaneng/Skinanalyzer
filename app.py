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
import io
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
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
# Lifting simulation (before / after with grid + contour shadow)
# Pure OpenCV/NumPy mesh warp - free, offline, no landmark model needed.
# The jaw/cheek tissue is pulled up and slightly inward by a smooth
# displacement field. The grid is drawn BEFORE warping, so its lines bend with
# the tissue and make the contour change visible.
# ----------------------------------------------------------------------------
RED = (235, 64, 52)
GREEN = (60, 220, 120)


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def lift_displacement(X, Y, box, strength):
    """Backward-mapping offsets (dx, dy) in pixels. strength is 0..1."""
    x, y, w, h = box
    cx = x + w / 2.0
    u = (X - cx) / (w / 2.0)          # -1..1 across the face
    v = (Y - y) / float(h)            # 0 = top of box, 1 = chin
    vert = _smooth((v - 0.45) / 0.35)                 # lower face only (eyes untouched)
    fade_v = 1 - _smooth((v - 1.0) / 0.25)            # fade out below the chin
    side = np.exp(-((np.abs(u) - 0.7) ** 2) / (2 * 0.4 ** 2))   # cheeks / jowls
    side = side * (1 - _smooth((np.abs(u) - 1.0) / 0.4))
    wgt = vert * fade_v * side
    dy = strength * 0.07 * h * wgt                              # tissue moves up
    dx = strength * 0.035 * w * wgt * np.sign(u) * np.minimum(np.abs(u), 1.0)  # and inward
    return dx, dy


def jaw_contour(box, n=60):
    """Approximate jawline from cheek to chin to cheek, as (x, y) points."""
    x, y, w, h = box
    cx = x + w / 2.0
    t = np.linspace(0, np.pi, n)
    c = np.cos(t)
    px = cx + np.sign(c) * np.abs(c) ** 1.35 * (w / 2.0) * 0.98
    py = y + 0.5 * h + 0.5 * h * np.sin(t)
    return np.stack([px, py], axis=1).astype(np.float32)


def forward_points(pts, box, strength):
    """Where contour points land after the warp (fixed-point inverse)."""
    o = pts.copy()
    for _ in range(6):
        dx, dy = lift_displacement(o[:, 0], o[:, 1], box, strength)
        o = pts - np.stack([dx, dy], axis=1)
    return o


def draw_grid(img, box, cells):
    """Grid spans exactly the box: top line = forehead, bottom line = chin."""
    x, y, w, h = box
    H, W = img.shape[:2]
    gx0, gx1 = max(x, 0), min(x + w, W - 1)
    gy0, gy1 = max(y, 0), min(y + h, H - 1)
    th = max(1, int(round(w / 220)))
    layer = img.copy()
    for xi in np.linspace(gx0, gx1, cells + 1):
        cv2.line(layer, (int(xi), int(gy0)), (int(xi), int(gy1)), (255, 255, 255), th, cv2.LINE_AA)
    for yi in np.linspace(gy0, gy1, cells + 1):
        cv2.line(layer, (int(gx0), int(yi)), (int(gx1), int(yi)), (255, 255, 255), th, cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.6, img, 0.4, 0)


def draw_poly(img, pts, color, th, dashed=False):
    p = pts.astype(np.int32)
    if not dashed:
        cv2.polylines(img, [p.reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)
        return
    for i in range(0, len(p) - 1, 2):
        cv2.line(img, tuple(map(int, p[i])), tuple(map(int, p[i + 1])), color, th, cv2.LINE_AA)


def simulate_lift(rgb, box, strength, show_grid=True, show_shadow=True, cells=12):
    """
    Returns (before_img, after_img, pct) cropped around the face.
    pct = largest jawline movement as % of face height.
    """
    H, W = rgb.shape[:2]
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx, dy = lift_displacement(X, Y, box, strength)

    base = draw_grid(rgb, box, cells) if show_grid else rgb.copy()
    # Warp the gridded image so grid lines bend with the tissue
    after = cv2.remap(base, (X + dx).astype(np.float32), (Y + dy).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    before = base.copy()

    pts = jaw_contour(box)
    new_pts = forward_points(pts, box, strength)
    th = max(2, int(round(box[2] / 150)))

    if show_shadow:
        # Soft shadow band between old and new jawline, drawn on the AFTER image
        shade = np.zeros((H, W), np.uint8)
        band = np.vstack([pts, new_pts[::-1]]).astype(np.int32)
        cv2.fillPoly(shade, [band.reshape(-1, 1, 2)], 255)
        shade = cv2.GaussianBlur(shade, (0, 0), max(2, box[2] / 60))
        alpha = (shade.astype(np.float32) / 255 * 0.55)[..., None]
        tint = np.array([20, 30, 60], np.float32)
        after = (after * (1 - alpha) + tint * alpha).astype(np.uint8)
        draw_poly(before, pts, RED, th)
        draw_poly(after, pts, RED, th, dashed=True)
        draw_poly(after, new_pts, GREEN, th)

    x, y, w, h = box
    x0, x1 = max(int(x - 0.35 * w), 0), min(int(x + 1.35 * w), W)
    y0, y1 = max(int(y - 0.25 * h), 0), min(int(y + 1.3 * h), H)
    pct = float(np.max(np.linalg.norm(new_pts - pts, axis=1)) / h * 100)
    return before[y0:y1, x0:x1], after[y0:y1, x0:x1], pct


# ----------------------------------------------------------------------------
# PDF export
# ----------------------------------------------------------------------------
def _rl_image(arr, width):
    img = Image.fromarray(arr)
    img.thumbnail((900, 900))
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=88)
    b.seek(0)
    return RLImage(b, width=width, height=width * img.height / img.width)


def build_pdf(a: Analysis, recs: list, client_name: str, lift: dict = None) -> bytes:
    text = consultation_text(a, recs)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Skin Consultation Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#1f3a34"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor("#1f3a34"), spaceBefore=10)
    body = ParagraphStyle("b", parent=styles["BodyText"], leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)

    story = [Paragraph("Skin Consultation Report", h1)]
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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a34")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6d2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f5")]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [t, Paragraph("Summary", h2), Paragraph(text["summary"], body),
              Paragraph(text["age_line"], body), Paragraph("Findings", h2)]
    story += [Paragraph("• " + f, body) for f in text["findings"]]
    story += [Paragraph("Recommended products", h2)]
    story += [Paragraph("• " + r, body) for r in text["recs"]]
    story += [Paragraph("Care guidance", h2)]
    story += [Paragraph("• " + tip, body) for tip in text["tips"]]
    if lift:
        story += [PageBreak(), Paragraph("Lifting simulation", h2)]
        pair = Table([[_rl_image(lift["before"], 82 * mm), _rl_image(lift["after"], 82 * mm)],
                      ["Before (original grid and jawline)", "After (simulated lift)"]],
                     colWidths=[86 * mm, 86 * mm])
        pair.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                  ("FONTSIZE", (0, 1), (-1, 1), 9)]))
        story += [pair, Spacer(1, 6), Paragraph(
            f"Simulated jawline lift at {lift['strength']}% intensity. Largest contour movement is "
            f"about {lift['pct']:.1f}% of face height. Red line = original jawline, green line = "
            f"simulated jawline, shaded band = area of change, grid lines bend with the lifted tissue.",
            body), Spacer(1, 4), Paragraph(
            "This is an illustrative visual simulation only. It is not a prediction or guarantee "
            "of any treatment result.", small)]
    story += [Spacer(1, 14), Paragraph(
        "This report is generated from an automated image analysis for cosmetic consultation only. "
        "It is not a medical diagnosis. Consult a qualified professional for skin conditions or concerns.",
        small)]
    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def score_card(col, title, value):
    with col:
        st.metric(title, f"{value:.0f}/100", label(value))
        st.progress(int(min(max(value, 0), 100)))


def main():
    st.set_page_config(page_title="Skin Analyzer", page_icon="🧴", layout="wide")
    st.title("🧴 Skin Analyzer")
    st.caption("Upload a well-lit, front-facing selfie for a cosmetic skin consultation.")

    with st.sidebar:
        st.header("Client details")
        name = st.text_input("Name (optional)")
        age = st.number_input("Your age", min_value=10, max_value=90, value=30, step=1)
        st.markdown("**Tips for best results**")
        st.markdown("- Natural, even light\n- No filters, makeup-free if possible\n"
                    "- Face the camera, hair off forehead\n- Remove glasses")

    source = st.radio("Photo source", ["Upload", "Camera"], horizontal=True)
    file = (st.file_uploader("Upload selfie", type=["jpg", "jpeg", "png", "webp"])
            if source == "Upload" else st.camera_input("Take a selfie"))

    if not file:
        st.info("Add a selfie to begin.")
        return

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

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.image(result.annotated, caption="Detected face and analysed skin region (green)",
                 use_container_width=True)
    with right:
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        score_card(c1, "Pigmentation", result.pigmentation)
        score_card(c2, "Texture", result.texture)
        score_card(c3, "Redness", result.redness)
        delta = result.skin_age - result.chronological_age
        st.metric("Skin age estimate", f"{result.skin_age} yrs",
                  f"{delta:+d} vs stated age", delta_color="inverse")
        st.caption("Higher scores = more visible concern. Skin age is a heuristic estimate.")
        for n in result.quality_notes:
            st.warning(n)

    st.divider()
    st.subheader("Consultation report")
    st.write(text["summary"])
    st.write(text["age_line"])
    st.markdown("**Findings**")
    for f in text["findings"]:
        st.markdown(f"- {f}")

    st.markdown("**Recommended products**")
    if recs:
        cols = st.columns(len(recs))
        for col, (pname, reasons) in zip(cols, recs):
            with col:
                st.success(f"**{pname}**")
                st.write(PRODUCTS[pname]["role"])
                st.caption("Addresses: " + ", ".join(reasons))
    else:
        st.write(text["recs"][0])

    st.markdown("**Care guidance**")
    for tip in text["tips"]:
        st.markdown(f"- {tip}")

    st.divider()
    st.subheader("Lifting result simulation")
    k1, k2, k3, k4 = st.columns(4)
    strength = k1.slider("Lift intensity", 0, 100, 60, help="Strength of the simulated jaw/cheek lift")
    cells = k2.slider("Grid density", 6, 20, 12)
    show_grid = k3.checkbox("Grid lines", value=True)
    show_shadow = k4.checkbox("Contour shadow", value=True)

    # --- Manual grid placement: forehead -> chin, left -> right -------------------
    H, W = rgb.shape[:2]
    fx, fy, fw, fh = result.face_box
    key = hashlib.md5(file.getvalue()).hexdigest()[:8]

    def _rng(lo, hi, default):
        lo = int(max(lo, 0))
        hi = int(max(hi, lo + 2))
        return lo, hi, int(np.clip(default, lo, hi))

    t_lo, t_hi, t_def = _rng(fy - 0.6 * fh, fy + 0.4 * fh, fy - 0.12 * fh)
    b_lo, b_hi, b_def = _rng(fy + 0.6 * fh, min(fy + 1.6 * fh, H - 1), fy + fh)
    l_lo, l_hi, l_def = _rng(fx - 0.5 * fw, fx + 0.5 * fw, fx)
    r_lo, r_hi, r_def = _rng(fx + 0.5 * fw, min(fx + 1.5 * fw, W - 1), fx + fw)
    keys = [f"top_{key}", f"bot_{key}", f"left_{key}", f"right_{key}"]

    with st.expander("Adjust grid to your face (forehead to chin)", expanded=True):
        st.caption("Move the top line to your hairline, the bottom line to your chin tip, "
                   "and the side lines to the edges of your cheeks. The picture below updates live.")
        g1, g2 = st.columns(2)
        top = g1.slider("Forehead (top line)", t_lo, t_hi, t_def, key=keys[0])
        bottom = g1.slider("Chin (bottom line)", b_lo, b_hi, b_def, key=keys[1])
        left = g2.slider("Left edge", l_lo, l_hi, l_def, key=keys[2])
        right = g2.slider("Right edge", r_lo, r_hi, r_def, key=keys[3])
        if st.button("Reset grid position"):
            for k in keys:
                st.session_state.pop(k, None)
            st.rerun()

    if bottom - top < 60 or right - left < 60:
        st.warning("Grid is too small – move the lines further apart. Using the detected face for now.")
        box = (fx, fy, fw, fh)
    else:
        box = (int(left), int(top), int(right - left), int(bottom - top))

    b_img, a_img, pct = simulate_lift(rgb, box, strength / 100, show_grid, show_shadow, cells)
    ic1, ic2 = st.columns(2)
    ic1.image(b_img, caption="Before – original grid and jawline (red)", use_container_width=True)
    ic2.image(a_img, caption="After – simulated lift: grid bends, green = new jawline, shaded = change",
              use_container_width=True)
    st.metric("Largest jawline movement", f"{pct:.1f}% of face height")
    st.caption("Illustrative simulation only – not a prediction or guarantee of any treatment result. "
               "Contour is estimated from the grid box you set, so a front-facing photo works best.")
    lift = {"before": b_img, "after": a_img, "pct": pct, "strength": strength}

    pdf = build_pdf(result, recs, name, lift)
    st.download_button("📄 Download PDF report", data=pdf,
                       file_name=f"skin_report_{datetime.now():%Y%m%d}.pdf",
                       mime="application/pdf", type="primary")
    st.caption("Cosmetic analysis only – not a medical diagnosis.")


if __name__ == "__main__":
    main()
