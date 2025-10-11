"""
advanced_forgery_generator.py

Generates realistic synthetic forgeries from RVL-CDIP images by applying a variety of tampering operations:
- text overlay/replacement
- copy-paste splicing (with optional Poisson seamless clone)
- signature/photo-like paste with affine/perspective transform
- local blur/smudge and warp
- JPEG compression artifacts and additive noise
- brightness/contrast/color jitter

Adjust PARAMETERS at top as needed.
"""

import os
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2

# -------------------------
# Configuration / params
# -------------------------
RVLCDIP_ROOT = r"E:\Thesis\rvl-cdip\images"   # input root (recursive)
OUTPUT_ROOT = r"E:\Thesis\forgery_dataset_realistic"
SAMPLE_SIZE = 500    # how many images to sample (set to total if you want everything)
SEED = 42

# Splits ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Forgery parameters (probabilities & sizes)
PROB_TEXT_OVERLAY = 0.45
PROB_COPY_PASTE = 0.6
PROB_SEAMLESS = 0.4            # if copy-paste, whether to use seamlessClone
PROB_SIGNATURE_PASTE = 0.25
PROB_LOCAL_BLUR = 0.5
PROB_COMPRESSION_NOISE = 0.6

MAX_PATCH_AREA = 0.08  # max patch area relative to image size for copy-paste
MIN_PATCH_AREA = 0.005

# Font path (try to provide a TTF path; fallback to default)
FONT_PATH = None  # e.g. r"C:\Windows\Fonts\arial.ttf" or leave None to use default PIL font

random.seed(SEED)
np.random.seed(SEED)


# -------------------------
# Utils
# -------------------------
def pil_to_cv(img_pil):
    arr = np.array(img_pil)
    # PIL uses RGB, OpenCV uses BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def add_jpeg_noise(pil_img, quality_range=(30, 95)):
    """Simulate compression artifacts by re-encoding to JPEG at low quality."""
    q = random.randint(*quality_range)
    from io import BytesIO
    b = BytesIO()
    pil_img.save(b, format="JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def add_gaussian_noise(pil_img, sigma=10):
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_brightness_contrast(pil_img, bright_range=(0.8, 1.3), contrast_range=(0.8, 1.3)):
    img = pil_img
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*bright_range))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast_range))
    return img


# -------------------------
# Forgery operations
# -------------------------
def text_overlay(pil_img):
    """Overlay or replace text in a random region (red text to simulate editing)."""
    img = pil_img.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # choose font
    font_size = max(12, int(min(w, h) * random.uniform(0.018, 0.045)))
    try:
        font = ImageFont.truetype(FONT_PATH or "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # choose random position and text
    text_candidates = ["Edited", "Name: John", "CONFIDENTIAL", "00/00/0000", "Amount: 0.00"]
    text = random.choice(text_candidates)

    # try to find an area to place text: random bounding box
    box_w = int(w * random.uniform(0.15, 0.5))
    box_h = int(h * random.uniform(0.03, 0.12))
    x = random.randint(0, max(0, w - box_w))
    y = random.randint(0, max(0, h - box_h))

    # draw semi-realistic overlay: white rectangle (simulate covering text) then write black/red text
    if random.random() < 0.5:
        # overlay a white patch then write text (simulate replacing)
        draw.rectangle([x, y, x + box_w, y + box_h], fill=(255, 255, 255))
        text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    else:
        text_color = (180, 10, 10)  # redish edited text

    draw.text((x + 5, y + 2), text, font=font, fill=text_color)
    return img


def copy_paste_splice(pil_img, seamless_prob=PROB_SEAMLESS):
    """Copy a random patch and paste it elsewhere. Optionally use Poisson blending."""
    img_cv = pil_to_cv(pil_img)
    h, w = img_cv.shape[:2]

    # select patch size
    total_area = h * w
    patch_area = int(total_area * random.uniform(MIN_PATCH_AREA, MAX_PATCH_AREA))
    # maintain aspect ratio roughly
    patch_w = int(np.sqrt(patch_area) * random.uniform(0.8, 1.2))
    patch_h = int(patch_area / max(1, patch_w))
    patch_w = min(max(10, patch_w), w // 2)
    patch_h = min(max(10, patch_h), h // 2)

    # pick source box
    sx = random.randint(0, w - patch_w)
    sy = random.randint(0, h - patch_h)
    src = img_cv[sy:sy + patch_h, sx:sx + patch_w].copy()

    # destination
    dx = random.randint(0, w - patch_w)
    dy = random.randint(0, h - patch_h)

    if random.random() < seamless_prob:
        # seamlessClone expects mask and center
        mask = 255 * np.ones(src.shape[:2], src.dtype)
        center = (dx + patch_w // 2, dy + patch_h // 2)
        try:
            img_cv = cv2.seamlessClone(src, img_cv, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # fallback to simple paste
            img_cv[dy:dy + patch_h, dx:dx + patch_w] = src
    else:
        # simple paste with possible opacity
        alpha = random.uniform(0.85, 1.0)
        dst_patch = img_cv[dy:dy + patch_h, dx:dx + patch_w]
        blended = cv2.addWeighted(dst_patch.astype(np.float32), 1 - alpha,
                                  src.astype(np.float32), alpha, 0).astype(np.uint8)
        img_cv[dy:dy + patch_h, dx:dx + patch_w] = blended

    return cv_to_pil(img_cv)


def signature_photo_paste(pil_img):
    """Copy a smaller region, apply rotations / perspective / blur and paste as a 'signature/photo'."""
    img_cv = pil_to_cv(pil_img)
    h, w = img_cv.shape[:2]

    # choose small patch
    patch_area = int(h * w * random.uniform(0.005, 0.02))
    patch_w = max(20, int(np.sqrt(patch_area)))
    patch_h = max(10, int(patch_area / max(1, patch_w)))

    sx = random.randint(0, w - patch_w)
    sy = random.randint(0, h - patch_h)
    src = img_cv[sy:sy + patch_h, sx:sx + patch_w].copy()

    # apply transforms: rotate, scale, perspective
    angle = random.uniform(-20, 20)
    scale = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((patch_w / 2, patch_h / 2), angle, scale)
    src = cv2.warpAffine(src, M, (patch_w, patch_h), borderMode=cv2.BORDER_REFLECT)

    # perspective warp
    if random.random() < 0.6:
        pts1 = np.float32([[0, 0], [patch_w - 1, 0], [0, patch_h - 1], [patch_w - 1, patch_h - 1]])
        delta = patch_w * 0.15
        pts2 = pts1 + np.float32([[random.uniform(-delta, delta), random.uniform(-delta, delta)] for _ in range(4)])
        mat = cv2.getPerspectiveTransform(pts1, pts2)
        src = cv2.warpPerspective(src, mat, (patch_w, patch_h), borderMode=cv2.BORDER_REFLECT)

    # optional blur to simulate signature ink bleed or softness
    if random.random() < 0.5:
        k = random.choice([3,5])
        src = cv2.GaussianBlur(src, (k, k), 0)

    # paste it somewhere
    dx = random.randint(0, w - patch_w)
    dy = random.randint(0, h - patch_h)

    # create mask for pasted content (non-white / non-background)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 3)

    roi = img_cv[dy:dy + patch_h, dx:dx + patch_w]
    try:
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        fg = cv2.bitwise_and(src, src, mask=mask)
        composed = cv2.add(bg, fg)
        img_cv[dy:dy + patch_h, dx:dx + patch_w] = composed
    except Exception:
        img_cv[dy:dy + patch_h, dx:dx + patch_w] = src

    return cv_to_pil(img_cv)


def local_blur_smudge(pil_img):
    """Apply local blur or smudge: choose a region and blur/smear it."""
    img_cv = pil_to_cv(pil_img)
    h, w = img_cv.shape[:2]

    patch_w = int(w * random.uniform(0.05, 0.3))
    patch_h = int(h * random.uniform(0.03, 0.25))
    x = random.randint(0, w - patch_w)
    y = random.randint(0, h - patch_h)

    # sometimes motion blur
    sub = img_cv[y:y + patch_h, x:x + patch_w].copy()
    if random.random() < 0.4:
        # gaussian blur
        k = random.choice([5,9,15])
        sub = cv2.GaussianBlur(sub, (k, k), 0)
    else:
        # motion blur: kernel
        size = random.choice([10, 15, 25])
        kernel = np.zeros((size, size))
        if random.random() < 0.5:
            kernel[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel[:, int((size - 1) / 2)] = np.ones(size)
        kernel = kernel / np.sum(kernel)
        sub = cv2.filter2D(sub, -1, kernel)

    img_cv[y:y + patch_h, x:x + patch_w] = sub
    return cv_to_pil(img_cv)


# -------------------------
# Overall forgery generator
# -------------------------
def make_forgery(pil_img):
    """Apply a sequence of forgery operations chosen randomly to create a realistic forged image."""
    img = pil_img.copy().convert("RGB")

    # Randomly apply text overlay
    if random.random() < PROB_TEXT_OVERLAY:
        img = text_overlay(img)

    # Copy-paste splice
    if random.random() < PROB_COPY_PASTE:
        img = copy_paste_splice(img, seamless_prob=PROB_SEAMLESS if random.random() < 0.8 else 0.0)

    # signature/photo paste
    if random.random() < PROB_SIGNATURE_PASTE:
        img = signature_photo_paste(img)

    # local blur / smudge
    if random.random() < PROB_LOCAL_BLUR:
        img = local_blur_smudge(img)

    # brightness/contrast jitter
    img = random_brightness_contrast(img)

    # compress / noise
    if random.random() < PROB_COMPRESSION_NOISE:
        img = add_jpeg_noise(img, quality_range=(30, 85))
        img = add_gaussian_noise(img, sigma=random.uniform(3, 16))

    return img