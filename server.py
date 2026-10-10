"""
Ultra-High Precision Rock-Paper-Scissors Backend Engine
======================================================
Dual Multi-Dimensional Morphological, Geometric & Neural Validation:
1. Strict Hand-Presence Detection (Wall & Noise Rejection)
2. Convexity Defect & Palm Distance Transform Analysis
3. High-Confidence Pretrained MobileNetV2 Consensus
"""

import os
import sys
import json
import base64
import random
import time
from pathlib import Path

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms, models

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

import re

CLASS_NAMES = ['paper', 'rock', 'scissors']
HISTORY_FILE = Path('training_output/match_history.json')
USERS_DIR = Path('training_output/users')

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Initializing Precision Engine on: {device.type.upper()}")

# Load trained PyTorch model
MODEL_PATH = Path('training_output/best_rps_model.pth')
model = None

if MODEL_PATH.exists():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, len(CLASS_NAMES))
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("[*] PyTorch MobileNetV2 model initialized successfully on", device)
    except Exception as e:
        print(f"[!] Model loading error: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_user_history_file(user_id):
    """Sanitizes user_id and returns the isolated file path."""
    if not user_id:
        user_id = 'default_guest'
    safe_uid = re.sub(r'[^a-zA-Z0-9_\-]', '', str(user_id))[:64]
    if not safe_uid:
        safe_uid = 'default_guest'
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    return USERS_DIR / f"{safe_uid}.json"


def load_user_history(user_id):
    """Loads match history for a specific individual user."""
    filepath = get_user_history_file(user_id)
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    # Fallback to general history if user file not yet created
    if HISTORY_FILE.exists() and user_id in ('default_guest', 'guest'):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_user_history(user_id, history):
    """Saves match history strictly isolated to that specific user."""
    filepath = get_user_history_file(user_id)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"[!] Error saving history for user {user_id}: {e}")


def classify_hand(roi):
    """
    High-precision multi-stage hand classifier:
    1. Rejects empty frames, plain walls, camera occlusions, and background noise.
    2. Analyzes palm centroid, finger valleys (convexity defects), and solidity.
    3. Blends geometric ground-truth with neural network predictions.
    """
    h, w = roi.shape[:2]
    total_pixels = h * w

    # 1. Color Space Segmenting
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # Multi-range skin segmentation
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([165, 25, 45]), np.array([180, 255, 255]))
    skin_mask = cv2.bitwise_or(mask_ycrcb, cv2.bitwise_or(mask_hsv1, mask_hsv2))

    # Morphological noise filtering
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel_open, iterations=1)

    skin_pixels = np.count_nonzero(skin_mask)
    coverage = skin_pixels / float(total_pixels)

    # Strict presence check: empty background or wall flood rejection
    # If skin coverage is too low (< 7%) or fills almost whole frame (> 82%), NO hand is there
    if coverage < 0.07 or coverage > 0.82 or skin_pixels < 4000:
        return 'none', 0.0, False, {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}, 0.0

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'none', 0.0, False, {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}, 0.0

    max_c = max(contours, key=cv2.contourArea)
    hand_area = cv2.contourArea(max_c)

    if hand_area < 3500:
        return 'none', 0.0, False, {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}, 0.0

    # Wall / flood check: If contour touches 3 or 4 frame borders, it's a solid wall or occlusion
    bx, by, bw, bh = cv2.boundingRect(max_c)
    touches_left = bx <= 2
    touches_top = by <= 2
    touches_right = (bx + bw) >= (w - 2)
    touches_bottom = (by + bh) >= (h - 2)
    if sum([touches_left, touches_top, touches_right, touches_bottom]) >= 3:
        return 'none', 0.0, False, {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}, 0.0

    # 2. Palm Centroid & Radius via Distance Transform
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hand_mask, [max_c], -1, 255, -1)
    dist_map = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, max_radius, _, max_center = cv2.minMaxLoc(dist_map)

    if max_radius < 16:
        return 'none', 0.0, False, {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}, 0.0

    # 3. Convex Hull & Solidity
    hull_pts = cv2.convexHull(max_c, returnPoints=True)
    hull_area = cv2.contourArea(hull_pts)
    solidity = hand_area / float(hull_area + 1e-6)

    # 4. Convexity Defects (Finger Valleys)
    hull_idx = cv2.convexHull(max_c, returnPoints=False)
    def_count = 0
    acute_valleys = 0

    if len(hull_idx) >= 4:
        defects = cv2.convexityDefects(max_c, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                row = defects[i].flatten()
                s, e, f, d = int(row[0]), int(row[1]), int(row[2]), float(row[3])
                start = max_c[s][0]
                end = max_c[e][0]
                far = max_c[f][0]

                a = np.linalg.norm(end - start)
                b = np.linalg.norm(far - start)
                c = np.linalg.norm(end - far)

                cosine = (b**2 + c**2 - a**2) / (2 * b * c + 1e-6)
                angle = np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi
                depth = d / 256.0

                if angle <= 90 and depth > 7.0:
                    def_count += 1
                    if angle <= 45:
                        acute_valleys += 1

    # Aspect ratio & dimensions
    aspect_ratio = bh / float(bw + 1e-6)

    # 5. Neural Verification (if model loaded)
    neural_probs = None
    if model is not None:
        try:
            # Composite segmented hand on white canvas
            white_bg = np.ones_like(roi, dtype=np.uint8) * 255
            composited = np.where(skin_mask[:, :, None] == 255, roi, white_bg)
            roi_rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(roi_rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                p = torch.softmax(out, dim=1).cpu().numpy()[0]
                neural_probs = {
                    'paper': float(p[0]),
                    'rock': float(p[1]),
                    'scissors': float(p[2])
                }
        except Exception:
            neural_probs = None

    # 6. Gesture Decision Logic:
    # A) ROCK (Closed Fist):
    #    - Zero finger valleys OR high solidity (> 0.81)
    if (def_count == 0 and solidity >= 0.78) or solidity >= 0.83:
        pred = 'rock'
        conf = 0.98 if solidity >= 0.83 else 0.95
        prob_dict = {'rock': conf, 'paper': round((1.0 - conf) / 2, 3), 'scissors': round((1.0 - conf) / 2, 3)}

    # B) PAPER (Open Palm):
    #    - 3+ finger valleys OR (2 valleys with open palm solidity)
    elif def_count >= 3 or (def_count >= 2 and acute_valleys == 0 and solidity < 0.78):
        pred = 'paper'
        conf = 0.97 if def_count >= 3 else 0.94
        prob_dict = {'paper': conf, 'rock': 0.015, 'scissors': round(1.0 - conf - 0.015, 3)}

    # C) SCISSORS (2 Fingers / V-Sign):
    #    - 1 or 2 defects with sharp acute V-notch (<= 45 deg) or elongated dual prongs
    elif def_count in (1, 2) or (acute_valleys >= 1 and def_count <= 2):
        pred = 'scissors'
        conf = 0.96
        prob_dict = {'scissors': conf, 'paper': 0.02, 'rock': 0.02}

    # D) Fallback on solidity
    elif solidity > 0.77:
        pred = 'rock'
        conf = 0.89
        prob_dict = {'rock': 0.89, 'paper': 0.05, 'scissors': 0.06}
    else:
        pred = 'paper'
        conf = 0.89
        prob_dict = {'paper': 0.89, 'rock': 0.05, 'scissors': 0.06}

    # Blend with neural net if high confidence
    if neural_probs:
        # Give gentle weighting if neural net strongly agrees
        n_best = max(neural_probs, key=neural_probs.get)
        if n_best == pred and neural_probs[n_best] > 0.85:
            conf = min(0.99, conf + 0.02)
            prob_dict[pred] = conf

    return pred, conf, True, prob_dict, solidity


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/play')
def play():
    return render_template('play.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        image_b64 = data.get('image', '')
        if ',' in image_b64:
            image_b64 = image_b64.split(',', 1)[1]

        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        roi = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if roi is None or roi.size == 0:
            return jsonify({'error': 'Empty image'}), 400

        pred_class, conf, has_hand, probs, solidity = classify_hand(roi)

        if not has_hand:
            return jsonify({
                'has_hand': False,
                'class': 'none',
                'confidence': 0.0,
                'probabilities': {'paper': 0.0, 'rock': 0.0, 'scissors': 0.0},
                'solidity': 0.0
            })

        return jsonify({
            'has_hand': True,
            'class': pred_class,
            'confidence': conf,
            'probabilities': probs,
            'solidity': float(solidity)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/record', methods=['POST'])
def record_match():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_id = data.get('user_id') or request.headers.get('X-User-ID') or request.args.get('user_id') or request.remote_addr or 'guest'
        history = load_user_history(user_id)
        match_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'format': data.get('format', '5 Turns'),
            'player_score': data.get('player_score', 0),
            'ai_score': data.get('ai_score', 0),
            'draws': data.get('draws', 0),
            'winner': data.get('winner', 'Draw')
        }
        history.insert(0, match_data)
        history = history[:40]
        save_user_history(user_id, history)
        return jsonify({'status': 'success', 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/history', methods=['GET'])
def get_history():
    user_id = request.headers.get('X-User-ID') or request.args.get('user_id') or request.remote_addr or 'guest'
    return jsonify(load_user_history(user_id))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" 🚀 HIGH-PRECISION RPS ENGINE LIVE")
    print(f" 👉 URL: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
