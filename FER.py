from flask import Flask, request, jsonify, send_from_directory, Response
from flask_mail import Mail, Message
import cv2
import os
import numpy as np
import base64
import json
from fer.fer import FER
from werkzeug.utils import secure_filename
import threading
import time
import sys

from database.db import (
    init_db,
    db_get_user, db_create_user, db_update_user_fields,
    db_get_session_user, db_create_session, db_delete_session, db_delete_sessions_by_email,
    db_set_pending_2fa, db_get_pending_2fa, db_delete_pending_2fa,
    db_set_pending_verification, db_get_pending_verification, db_delete_pending_verification,
    db_set_pending_reset, db_get_pending_reset, db_delete_pending_reset_by_token,
    db_save_archive, db_get_archive_by_user, db_get_archive_entry,
    db_delete_archive_entry, db_update_archive_entry, db_update_archive_image_path,
)

# Compatibility patch: torchvision >= 0.16 removed functional_tensor,
# but basicsr (used by GFPGAN / Real-ESRGAN) still imports it.
try:
    import torchvision.transforms.functional_tensor  # noqa: F401  # type: ignore[import]
except ImportError:
    import torchvision.transforms.functional as _F
    sys.modules['torchvision.transforms.functional_tensor'] = _F

# --- Face enhancement models (lazy-loaded on first use) ---
_gfpgan_restorer = None
_realesrgan_upsampler = None

def get_gfpgan():
    global _gfpgan_restorer
    if _gfpgan_restorer is None:
        try:
            from gfpgan import GFPGANer
            model_path = os.path.join(BASE_DIR, 'models', 'GFPGANv1.4.pth')
            _gfpgan_restorer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        except Exception as e:
            raise RuntimeError(f'GFPGAN load error: {e}')
    return _gfpgan_restorer

def get_realesrgan():
    global _realesrgan_upsampler
    if _realesrgan_upsampler is None:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model_path = os.path.join(BASE_DIR, 'models', 'RealESRGAN_x4plus.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            _realesrgan_upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
        except Exception as e:
            raise RuntimeError(f'Real-ESRGAN load error: {e}')
    return _realesrgan_upsampler

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database (creates tables + migrates JSON data if needed)
try:
    init_db()
except Exception as _db_err:
    print(f'[DB] ERROR during init: {_db_err}')


# Flask-Mail — loaded from email_config.json (kept outside DB for security)
def _configure_mail(application):
    cfg_path = os.path.join(BASE_DIR, 'database', 'email_config.json')
    if not os.path.exists(cfg_path):
        return
    with open(cfg_path) as _f:
        _cfg = json.load(_f)
    application.config['MAIL_SERVER']   = _cfg.get('smtp_host', 'smtp.gmail.com')
    application.config['MAIL_PORT']     = int(_cfg.get('smtp_port', 587))
    application.config['MAIL_USE_SSL']  = bool(_cfg.get('smtp_ssl', False))
    application.config['MAIL_USE_TLS']  = not bool(_cfg.get('smtp_ssl', False))
    application.config['MAIL_USERNAME'] = _cfg.get('smtp_username', '')
    application.config['MAIL_PASSWORD'] = _cfg.get('smtp_password', '')
    application.config['MAIL_DEFAULT_SENDER'] = (
        _cfg.get('from_name', 'FER System'),
        _cfg.get('from_address', _cfg.get('smtp_username', ''))
    )
    if _cfg.get('base_url'):
        application.config['BASE_URL'] = _cfg['base_url'].rstrip('/')

_configure_mail(app)
mail = Mail(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)

# Initialize face detector and emotion detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
emotion_detector = FER(mtcnn=True)

# Camera state
camera_active = False
camera_thread = None
camera_frame = None
camera_frame_clean = None  # Clean frame without emotion overlays
camera_lock = threading.Lock()


def apply_random_transformations(face_img):
    """Apply transformations for better emotion detection"""
    import random
    
    if random.choice([True, False]):
        max_shift = 20
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        face_img = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
    
    if random.choice([True, False]):
        brightness_factor = random.uniform(0.5, 1.5)
        face_img = cv2.convertScaleAbs(face_img, alpha=brightness_factor, beta=0)
    
    return face_img


@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'templates'), 'combined.html')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/enhance_faces', methods=['POST'])
def enhance_faces():
    """Enhance detected face crops using GFPGAN or Real-ESRGAN"""
    data = request.json
    filepath = data.get('filepath')
    faces = data.get('faces', [])
    model_name = data.get('model', 'gfpgan')  # 'gfpgan' or 'realesrgan'

    if not filepath or not faces:
        return jsonify({'error': 'Missing data'}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404

    enhanced_faces = []

    try:
        if model_name == 'gfpgan':
            restorer = get_gfpgan()
        else:
            upsampler = get_realesrgan()
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500

    if model_name == 'gfpgan':
        # GFPGAN: process the full image once and paste results back in place.
        # This avoids rotation/alignment artifacts that appear when passing
        # individual face crops (GFPGAN re-aligns each crop to canonical pose).
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _, _, restored_img = restorer.enhance(
                img_rgb,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            if restored_img is not None:
                restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            else:
                restored_bgr = img
        except Exception as e:
            print(f'GFPGAN full-image enhancement error: {e}')
            restored_bgr = img

        for face in faces:
            face_id = face['id']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            enhanced = restored_bgr[y:y+h, x:x+w].copy()
            enh_filename = f'enhanced_gfpgan_{os.path.splitext(filepath)[0]}_face_{face_id}.jpg'
            enh_path = os.path.join(app.config['UPLOAD_FOLDER'], enh_filename)
            cv2.imwrite(enh_path, enhanced)
            enhanced_faces.append({'id': face_id, 'enhanced_crop': f'/uploads/{enh_filename}'})

    else:
        # Real-ESRGAN: process each face crop individually (no alignment step)
        for face in faces:
            face_id = face['id']
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            face_crop = img[y:y+h, x:x+w].copy()
            try:
                enhanced, _ = upsampler.enhance(face_crop, outscale=2)
            except Exception as e:
                print(f'Real-ESRGAN enhancement error for face {face_id}: {e}')
                enhanced = face_crop
            enh_filename = f'enhanced_realesrgan_{os.path.splitext(filepath)[0]}_face_{face_id}.jpg'
            enh_path = os.path.join(app.config['UPLOAD_FOLDER'], enh_filename)
            cv2.imwrite(enh_path, enhanced)
            enhanced_faces.append({'id': face_id, 'enhanced_crop': f'/uploads/{enh_filename}'})

    return jsonify({'success': True, 'enhanced_faces': enhanced_faces})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detect faces"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Read image
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # Detect faces
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(20, 20)
    )
    
    # Extract and save cropped faces
    detected_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y+h, x:x+w]
        
        # Save cropped face
        face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
        face_filepath = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        cv2.imwrite(face_filepath, face_img)
        
        detected_faces.append({
            'id': i,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'cropped_face': f'/uploads/{face_filename}'
        })
    
    return jsonify({
        'success': True,
        'filepath': filename,
        'faces': detected_faces
    })


@app.route('/analyze_faces', methods=['POST'])
def analyze_faces():
    """Analyze emotions for selected faces"""
    data = request.json
    filepath = data.get('filepath')
    accepted_faces = data.get('accepted_faces', [])

    if not filepath or not accepted_faces:
        return jsonify({'error': 'Missing data'}), 400

    # Read original image (ensure we don't read the analyzed version)
    # Remove 'analyzed_' prefix if present to get the original file
    original_filepath = filepath.replace('analyzed_', '', 1)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filepath)
    img = cv2.imread(img_path)

    if img is None:
        return jsonify({'error': 'Image not found'}), 404

    # Create a fresh copy of the image to draw on
    img = img.copy()
    img_height, img_width = img.shape[:2]

    # Calculate font scale based on image size (larger images = larger text)
    base_scale = min(img_width, img_height) / 1000.0
    font_scale = max(0.4, min(1.2, base_scale))  # Between 0.4 and 1.2
    font_thickness = max(1, int(font_scale * 2))
    outline_thickness = max(2, int(font_scale * 3))
    line_spacing = int(25 * font_scale)

    results = []

    for face in accepted_faces:
        face_id = face['id']
        x, y, w, h = face['x'], face['y'], face['w'], face['h']

        # Use enhanced crop if available, otherwise crop from original image
        enhanced_crop_path = face.get('enhanced_crop', '')
        if enhanced_crop_path:
            crop_filename = enhanced_crop_path.lstrip('/').replace('uploads/', '', 1)
            crop_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(crop_filename))
            face_img = cv2.imread(crop_full_path)
            if face_img is None:
                face_img = img[y:y+h, x:x+w]
        else:
            face_img = img[y:y+h, x:x+w]

        # Analyze emotions with transformations for better accuracy
        transformed_emotions = []
        if face_img is None or face_img.size == 0:
            continue
        for _ in range(10):
            transformed_face = apply_random_transformations(face_img.copy())
            if transformed_face is None or transformed_face.size == 0:
                transformed_face = face_img.copy()
            face_img_rgb = cv2.cvtColor(transformed_face, cv2.COLOR_BGR2RGB)
            detected_emotions = emotion_detector.detect_emotions(face_img_rgb)

            if detected_emotions:
                emotion_scores = detected_emotions[0]["emotions"]
                transformed_emotions.append(emotion_scores)
            else:
                transformed_emotions.append({"neutral": 1.0})

        # Average emotions
        avg_emotions = {
            emotion: 0 for emotion in
            ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        }

        for emotions in transformed_emotions:
            for emotion, score in emotions.items():
                avg_emotions[emotion] += score

        num_transforms = len(transformed_emotions)
        for emotion in avg_emotions:
            avg_emotions[emotion] = (avg_emotions[emotion] / num_transforms) * 100

        # Draw rectangle with scaled thickness
        rect_thickness = max(2, int(font_scale * 3))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), rect_thickness)

        # Prepare emotion labels
        label_texts = [f"ID: {face_id}"]
        for emotion, score in avg_emotions.items():
            label_texts.append(f"{emotion.capitalize()}: {score:.1f}%")

        # Calculate total height needed for all labels
        total_label_height = len(label_texts) * line_spacing

        # Strategy: Try positions in order of preference
        # 1. Above face
        # 2. Below face
        # 3. To the right of face
        # 4. Inside face at top

        label_start_y = None
        margin = 5

        # Try above face
        if y - total_label_height - margin >= 0:
            label_start_y = y - total_label_height + line_spacing
        # Try below face
        elif y + h + total_label_height + margin <= img_height:
            label_start_y = y + h + line_spacing
        # Try inside face at top (last resort)
        else:
            label_start_y = y + line_spacing
            # Limit labels to fit within face height
            max_lines = (h - margin) // line_spacing
            if max_lines < len(label_texts):
                label_texts = label_texts[:max(1, max_lines)]

        # Draw each label
        for line_num, label_text in enumerate(label_texts):
            label_y = label_start_y + (line_num * line_spacing)

            # Clamp Y to image bounds
            label_y = max(line_spacing, min(label_y, img_height - margin))

            # Ensure label x position keeps text within image bounds
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thickness)[0]
            label_x = x

            # Adjust x if text would go beyond right edge
            if label_x + text_size[0] + margin > img_width:
                label_x = img_width - text_size[0] - margin

            # Ensure x is not negative
            label_x = max(margin, label_x)

            # Draw text with outline for better visibility
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(img, label_text, (label_x + dx, label_y + dy),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                           outline_thickness, cv2.LINE_AA)
            cv2.putText(img, label_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                       font_thickness, cv2.LINE_AA)

        results.append({
            'face_id': face_id,
            'emotions': avg_emotions,
            'coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
    
    base_name = os.path.splitext(original_filepath)[0]
    extension = os.path.splitext(original_filepath)[1]

    # Save output image with time-based suffix (HHMMSScc) to ensure uniqueness
    from datetime import datetime
    dt = datetime.now()
    timestamp = dt.strftime('%H%M%S') + f"{dt.microsecond // 10000:02d}"
    output_filename = f"analyzed_{base_name}_{timestamp}{extension}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, img)

    return jsonify({
        'success': True,
        'results': results,
        'output_image': output_filename
    })


@app.route('/save_results', methods=['POST'])
def save_results():
    """Save analysis results to archive — requires authentication"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'error': 'You must be logged in to save results'}), 401
    u = db_get_session_user(token)
    if not u:
        return jsonify({'error': 'Invalid session. Please log in again.'}), 401

    body = request.json
    image_path = body.get('image_path')
    emotions = body.get('emotions', [])

    faces = {
        str(e['face_id']): {
            'emotions': e['emotions'],
            'coordinates': e.get('coordinates', {})
        } for e in emotions
    }
    db_save_archive(f'/uploads/{image_path}', time.time(), u['email'], faces)
    return jsonify({'success': True})


@app.route('/get_archive', methods=['GET'])
def get_archive():
    """Get archive entries – requires authentication, returns only own entries"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'success': True, 'archive': []})
    u = db_get_session_user(token)
    if not u:
        return jsonify({'success': True, 'archive': []})

    archive = db_get_archive_by_user(u['email'])
    return jsonify({'success': True, 'archive': archive})


@app.route('/load_archive_image', methods=['POST'])
def load_archive_image():
    """Load analyzed image from archive with emotions"""
    data = request.json
    image_path = data.get('image_path')
    emotions_data = data.get('emotions', {})

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    # Remove /uploads/ prefix if present to get just the filename
    if image_path.startswith('/uploads/'):
        filename = image_path.replace('/uploads/', '', 1)
    else:
        filename = os.path.basename(image_path)

    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(full_path):
        print(f"Archive image not found: {full_path}")
        return jsonify({'error': f'Image not found: {filename}'}), 404

    print(f"Loading archive image: {full_path}")
    # Return analyzed image path and emotions for display
    return jsonify({
        'success': True,
        'analyzed_image': image_path,
        'emotions': emotions_data
    })


@app.route('/delete_archive_entry', methods=['POST'])
def delete_archive_entry():
    """Delete an entry from the archive"""
    data = request.json
    image_path = data.get('image_path')

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    found = db_delete_archive_entry(image_path)
    if not found:
        return jsonify({'error': 'Entry not found in archive'}), 404

    # Optionally delete the associated image file
    try:
        filename = os.path.basename(image_path)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(full_path):
            os.remove(full_path)
    except OSError as e:
        print(f"Error deleting image file: {e}")

    return jsonify({'success': True})


@app.route('/update_archive_entry', methods=['POST'])
def update_archive_entry():
    """Update an existing archive entry (emotions or name)"""
    data = request.json
    image_path = data.get('image_path')
    emotions = data.get('emotions')
    custom_name = data.get('custom_name')

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    new_faces = None
    if emotions:
        # Need existing coordinates — fetch from DB first
        entry = db_get_archive_entry(image_path)
        existing_faces = entry.get('faces', {}) if entry else {}

        new_faces = {}
        for e in emotions:
            face_id = str(e['face_id'])
            existing_coords = {}
            if face_id in existing_faces:
                ef = existing_faces[face_id]
                if isinstance(ef, dict) and 'coordinates' in ef:
                    existing_coords = ef['coordinates']
            new_faces[face_id] = {
                'emotions': e['emotions'],
                'coordinates': e.get('coordinates', existing_coords)
            }

    found = db_update_archive_entry(image_path, faces=new_faces, custom_name=custom_name)
    if not found:
        return jsonify({'error': 'Entry not found in archive'}), 404

    return jsonify({'success': True})


@app.route('/update_image_emotions', methods=['POST'])
def update_image_emotions():
    """Regenerate analyzed image with updated emotions"""
    data = request.json
    image_path = data.get('image_path')
    emotions_data = data.get('emotions', [])

    if not image_path or not emotions_data:
        return jsonify({'error': 'Missing image path or emotions'}), 400

    # Extract the filename - handle both /uploads/filename and just filename
    if image_path.startswith('/uploads/'):
        filename = image_path.replace('/uploads/', '', 1)
    else:
        filename = image_path

    # Get the original image by removing analyzed_ prefix and timestamp
    # Pattern: analyzed_originalname_timestamp.ext -> originalname.ext
    import re
    # Remove 'analyzed_' prefix and the last timestamp (8-digit HHMMSScc or legacy 13-digit ms)
    original_filename = re.sub(r'^analyzed_(.+)_(\d{8}|\d{13})(\.\w+)$', r'\1\3', filename)

    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    # If original doesn't exist, try using the analyzed image itself (remove overlays not possible, but better than error)
    if not os.path.exists(original_path):
        print(f"Original image not found: {original_path}")
        # Try to use the current analyzed image as base
        analyzed_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(analyzed_path):
            print(f"Analyzed image also not found: {analyzed_path}")
            return jsonify({'error': f'Image not found: {filename}'}), 404
        print(f"Using analyzed image as base: {analyzed_path}")
        # Use analyzed image - not ideal but prevents errors
        img = cv2.imread(analyzed_path)
    else:
        # Read original image
        img = cv2.imread(original_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500

    img_height, img_width = img.shape[:2]

    # Calculate font scale based on image size
    base_scale = min(img_width, img_height) / 1000.0
    font_scale = max(0.4, min(1.2, base_scale))
    font_thickness = max(1, int(font_scale * 2))
    outline_thickness = max(2, int(font_scale * 3))
    line_spacing = int(25 * font_scale)

    # Load face coordinates from archive
    face_coordinates = {}
    _archive_entry = db_get_archive_entry(image_path)
    if _archive_entry:
        for face_id_str, face_data in _archive_entry.get('faces', {}).items():
            if isinstance(face_data, dict) and 'coordinates' in face_data:
                face_coordinates[int(face_id_str)] = face_data['coordinates']

    # Draw rectangles and labels for each face
    for emotion_entry in emotions_data:
        face_id = emotion_entry['face_id']
        emotions = emotion_entry['emotions']

        # Try to get coordinates from emotion_entry first (new analysis), then from archive
        if 'coordinates' in emotion_entry:
            coords = emotion_entry['coordinates']
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
        elif face_id in face_coordinates:
            coords = face_coordinates[face_id]
            x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
        else:
            # Fallback: detect faces (should rarely happen)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            if face_id < len(faces):
                x, y, w, h = faces[face_id]
            else:
                continue

        # Draw rectangle
        rect_thickness = max(2, int(font_scale * 3))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), rect_thickness)

        # Prepare emotion labels
        label_texts = [f"ID: {face_id}"]
        for emotion, score in emotions.items():
            label_texts.append(f"{emotion.capitalize()}: {score:.1f}%")

        # Calculate total height needed for all labels
        total_label_height = len(label_texts) * line_spacing

        # Determine label position
        label_start_y = None
        margin = 5

        # Try above face
        if y - total_label_height - margin >= 0:
            label_start_y = y - total_label_height + line_spacing
        # Try below face
        elif y + h + total_label_height + margin <= img_height:
            label_start_y = y + h + line_spacing
        # Try inside face at top
        else:
            label_start_y = y + line_spacing
            max_lines = (h - margin) // line_spacing
            if max_lines < len(label_texts):
                label_texts = label_texts[:max(1, max_lines)]

        # Draw each label
        for line_num, label_text in enumerate(label_texts):
            label_y = label_start_y + (line_num * line_spacing)
            label_y = max(line_spacing, min(label_y, img_height - margin))

            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, font_thickness)[0]
            label_x = x

            if label_x + text_size[0] + margin > img_width:
                label_x = img_width - text_size[0] - margin

            label_x = max(margin, label_x)

            # Draw text with outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(img, label_text, (label_x + dx, label_y + dy),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                           outline_thickness, cv2.LINE_AA)
            cv2.putText(img, label_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                       font_thickness, cv2.LINE_AA)

    # Save updated image with new timestamp
    timestamp = int(time.time() * 1000)
    base_name = os.path.splitext(original_filename)[0]
    extension = os.path.splitext(original_filename)[1]
    new_filename = f"analyzed_{base_name}_{timestamp}{extension}"
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    cv2.imwrite(new_path, img)

    # Delete old analyzed image
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(old_path) and old_path != new_path:
        try:
            os.remove(old_path)
        except OSError:
            pass

    # Update archive if this image is in archive
    if _archive_entry:
        existing_faces = _archive_entry.get('faces', {})
        new_faces = {}
        for e in emotions_data:
            face_id = str(e['face_id'])
            existing_coords = {}
            if face_id in existing_faces:
                ef = existing_faces[face_id]
                if isinstance(ef, dict) and 'coordinates' in ef:
                    existing_coords = ef['coordinates']
            new_faces[face_id] = {
                'emotions': e['emotions'],
                'coordinates': e.get('coordinates', existing_coords)
            }
        db_update_archive_image_path(
            _archive_entry['image_path'],
            f'/uploads/{new_filename}',
            new_faces
        )

    return jsonify({
        'success': True,
        'updated_image': f'/uploads/{new_filename}'
    })


# Camera stream functions
def camera_worker():
    """Background thread for camera capture"""
    global camera_active, camera_frame, camera_frame_clean

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_active:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Store clean frame without overlays
        clean_frame = frame.copy()

        # Detect faces and emotions
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        # Process each face
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue

            try:
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                detected_emotions = emotion_detector.detect_emotions(face_img_rgb)

                if detected_emotions:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    emotions = detected_emotions[0]["emotions"]

                    # Draw emotion labels
                    for line_num, (emotion, score) in enumerate(emotions.items()):
                        label_text = f"{emotion.capitalize()}: {score*100:.1f}%"
                        label_y = y + h + 20 + (line_num * 20)

                        # Draw with outline
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            cv2.putText(frame, label_text, (x + dx, label_y + dy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label_text, (x, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        # Update shared frames
        with camera_lock:
            camera_frame = frame.copy()  # Frame with overlays for display
            camera_frame_clean = clean_frame.copy()  # Clean frame for capture

        time.sleep(0.033)  # ~30 FPS

    video_capture.release()


@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera stream"""
    global camera_active, camera_thread
    
    if camera_active:
        return jsonify({'success': True, 'message': 'Camera already active'})
    
    camera_active = True
    camera_thread = threading.Thread(target=camera_worker)
    camera_thread.start()
    
    return jsonify({'success': True})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    global camera_active
    camera_active = False
    
    if camera_thread:
        camera_thread.join(timeout=2)
    
    return jsonify({'success': True})


@app.route('/camera_stream')
def camera_stream():
    """Stream camera frames as MJPEG"""
    def generate():
        while camera_active:
            with camera_lock:
                if camera_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', camera_frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Capture current camera frame and process it"""
    global camera_frame_clean

    with camera_lock:
        if camera_frame_clean is None:
            return jsonify({'error': 'No frame available'}), 400

        frame = camera_frame_clean.copy()

    # Validate frame
    if frame is None or frame.size == 0:
        return jsonify({'error': 'Invalid frame captured'}), 400

    # Save captured clean frame
    import time
    timestamp = int(time.time())
    filename = f'camera_capture_{timestamp}.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, frame)

    # Detect faces using a fresh classifier instance to avoid threading issues
    try:
        # Create a new classifier for this thread to avoid conflicts
        local_face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = local_face_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
    except cv2.error as e:
        print(f"Error detecting faces: {e}")
        return jsonify({'error': 'Error detecting faces in frame'}), 500

    detected_faces = []
    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame[y:y+h, x:x+w]
        face_filename = f'camera_capture_{timestamp}_face_{i}.jpg'
        face_filepath = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
        cv2.imwrite(face_filepath, face_img)

        detected_faces.append({
            'id': i,
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'cropped_face': f'/uploads/{face_filename}'
        })

    return jsonify({
        'success': True,
        'filepath': filename,
        'faces': detected_faces
    })


# ============================================================
# AUTH HELPERS
# ============================================================
import uuid
import random
from werkzeug.security import generate_password_hash, check_password_hash


def send_email(to_address, subject, body_html):
    with app.app_context():
        msg = Message(subject, recipients=[to_address], html=body_html)
        mail.send(msg)


import re as _re
def _validate_password(pw):
    """Returns error string or None if password is valid."""
    if len(pw) < 8:
        return 'Password must be at least 8 characters'
    if not _re.search(r'[A-Z]', pw):
        return 'Password must contain at least one uppercase letter'
    if not _re.search(r'[0-9]', pw):
        return 'Password must contain at least one digit'
    if not _re.search(r'[^A-Za-z0-9]', pw):
        return 'Password must contain at least one special character'
    return None


# ============================================================
# AUTH ENDPOINTS
# ============================================================

def _send_verification_code(email):
    """Generate and send a 6-digit verification code. Invalidates previous codes."""
    code = f'{random.randint(0, 999999):06d}'
    db_set_pending_verification(email, code, time.time() + 900)  # 15 min
    send_email(email, 'Verify your email – FER System',
               f'<p>Your verification code is:</p>'
               f'<h1 style="letter-spacing:8px;font-family:monospace">{code}</h1>'
               f'<p>This code is valid for <strong>15 minutes</strong>.</p>'
               f'<p>If you did not create an account, please ignore this email.</p>')
    return code


@app.route('/auth/register', methods=['POST'])
def auth_register():
    body = request.json
    email = (body.get('email') or '').strip().lower()
    password = body.get('password') or ''
    username = (body.get('username') or '').strip()
    avatar = body.get('avatar') or ''

    if not email or '@' not in email:
        return jsonify({'error': 'Invalid email'}), 400
    pw_err = _validate_password(password)
    if pw_err:
        return jsonify({'error': pw_err}), 400
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    existing = db_get_user(email)
    if existing and existing.get('verified'):
        return jsonify({'error': 'Email already registered'}), 409

    if existing:
        # Unverified account — update username/avatar and resend code
        updates = {'username': username}
        if avatar:
            updates['avatar'] = avatar
        db_update_user_fields(email, **updates)
    else:
        db_create_user(email, generate_password_hash(password), username, avatar)

    try:
        _send_verification_code(email)
    except Exception as e:
        if not existing:
            # Remove user we just created if email failed
            from database.db import get_conn
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("DELETE FROM users WHERE email=%s", (email,))
            conn.commit(); cur.close(); conn.close()
        return jsonify({'error': f'Failed to send email: {e}'}), 500

    return jsonify({'success': True, 'message': 'Verification code sent to your email'})


@app.route('/auth/verify_registration', methods=['POST'])
def auth_verify_registration():
    body = request.json
    email = (body.get('email') or '').strip().lower()
    code = (body.get('code') or '').strip()

    entry = db_get_pending_verification(email)
    if not entry:
        return jsonify({'error': 'No pending verification for this email'}), 400
    if time.time() > entry['expires']:
        db_delete_pending_verification(email)
        return jsonify({'error': 'Code has expired. Please register again.'}), 400
    if entry['code'] != code:
        return jsonify({'error': 'Invalid code'}), 401

    user = db_get_user(email)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    db_update_user_fields(email, verified=1)
    db_delete_pending_verification(email)

    try:
        send_email(email, 'Welcome to FER System!',
                   f'<p>Your account has been verified. You can now log in.</p>'
                   f'<p>Welcome to FER System!</p>')
    except Exception:
        pass  # Welcome email is non-critical

    return jsonify({'success': True, 'message': 'Email verified! You can now log in.'})


@app.route('/auth/resend_verification', methods=['POST'])
def auth_resend_verification():
    body = request.json
    email = (body.get('email') or '').strip().lower()

    user = db_get_user(email)
    if not user:
        return jsonify({'error': 'Email not found'}), 404
    if user.get('verified'):
        return jsonify({'error': 'Account already verified'}), 400

    try:
        _send_verification_code(email)
    except Exception as e:
        return jsonify({'error': f'Failed to send email: {e}'}), 500

    return jsonify({'success': True, 'message': 'New verification code sent'})


@app.route('/auth/login', methods=['POST'])
def auth_login():
    body = request.json
    email = (body.get('email') or '').strip().lower()
    password = body.get('password') or ''

    user = db_get_user(email)
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid email or password'}), 401
    if not user.get('verified'):
        return jsonify({'error': 'Email not verified. Check your inbox.'}), 403

    code = f'{random.randint(0, 999999):06d}'
    db_set_pending_2fa(email, code, time.time() + 600)  # 10 min

    try:
        send_email(email, 'Your login code – FER App',
                   f'<p>Your verification code is:</p>'
                   f'<h1 style="letter-spacing:8px">{code}</h1>'
                   f'<p>This code is valid for <strong>10 minutes</strong>.</p>'
                   f'<p>If you did not request this, ignore this email.</p>')
    except Exception as e:
        return jsonify({'error': f'Failed to send 2FA email: {e}'}), 500

    return jsonify({'success': True, 'message': '2FA code sent to your email'})


@app.route('/auth/verify_2fa', methods=['POST'])
def auth_verify_2fa():
    body = request.json
    email = (body.get('email') or '').strip().lower()
    code = (body.get('code') or '').strip()

    entry = db_get_pending_2fa(email)
    if not entry:
        return jsonify({'error': 'No pending verification for this email'}), 400
    if time.time() > entry['expires']:
        db_delete_pending_2fa(email)
        return jsonify({'error': 'Code has expired. Please log in again.'}), 400
    if entry['code'] != code:
        return jsonify({'error': 'Invalid code'}), 401

    db_delete_pending_2fa(email)
    session_token = str(uuid.uuid4())
    db_create_session(session_token, email)

    user = db_get_user(email)
    return jsonify({
        'success': True,
        'token': session_token,
        'email': user['email'],
        'username': user.get('username', ''),
        'avatar': user.get('avatar', '')
    })


@app.route('/auth/me', methods=['GET'])
def auth_me():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'logged_in': False}), 200
    user = db_get_session_user(token)
    if not user:
        return jsonify({'logged_in': False}), 200
    return jsonify({
        'logged_in': True,
        'email': user['email'],
        'username': user.get('username', ''),
        'avatar': user.get('avatar', '')
    })


@app.route('/auth/update_profile', methods=['POST'])
def auth_update_profile():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'error': 'Not authenticated'}), 401
    user = db_get_session_user(token)
    if not user:
        return jsonify({'error': 'Invalid session'}), 401

    body = request.json
    updates = {}
    if 'username' in body:
        username = (body['username'] or '').strip()
        if not username:
            return jsonify({'error': 'Username cannot be empty'}), 400
        updates['username'] = username
    if 'avatar' in body:
        updates['avatar'] = body['avatar']
    if 'new_password' in body:
        current_pw = body.get('current_password') or ''
        new_pw = body.get('new_password') or ''
        if not check_password_hash(user['password_hash'], current_pw):
            return jsonify({'error': 'Current password is incorrect'}), 401
        pw_err = _validate_password(new_pw)
        if pw_err:
            return jsonify({'error': pw_err}), 400
        updates['password_hash'] = generate_password_hash(new_pw)

    if updates:
        db_update_user_fields(user['email'], **updates)

    updated = db_get_user(user['email'])
    return jsonify({
        'success': True,
        'username': updated.get('username', ''),
        'avatar': updated.get('avatar', '')
    })


@app.route('/auth/forgot_password', methods=['POST'])
def auth_forgot_password():
    body = request.json
    email = (body.get('email') or '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    user = db_get_user(email)
    if not user:
        print(f'[forgot_password] No user found for: {email}')
        return jsonify({'success': True, 'message': 'If this email exists, a reset link has been sent.'})
    if not user.get('verified'):
        print(f'[forgot_password] User {email} is not verified')
        return jsonify({'success': True, 'message': 'If this email exists, a reset link has been sent.'})

    reset_token = str(uuid.uuid4())
    db_set_pending_reset(email, reset_token, time.time() + 3600)  # 1 hour

    base_url = app.config.get('BASE_URL', request.host_url.rstrip('/'))
    reset_link = f'{base_url}/?reset_token={reset_token}'
    print(f'[forgot_password] Sending reset link to {email}: {reset_link}')
    try:
        send_email(email, 'Reset your FER System password',
                   f'<p>You requested a password reset.</p>'
                   f'<p><a href="{reset_link}" style="font-size:16px;">Click here to reset your password</a></p>'
                   f'<p>This link is valid for <strong>1 hour</strong>.</p>'
                   f'<p>If you did not request this, you can ignore this email.</p>')
        print(f'[forgot_password] Email sent successfully to {email}')
    except Exception as e:
        print(f'[forgot_password] Failed to send email: {e}')
        return jsonify({'error': f'Failed to send email: {e}'}), 500

    return jsonify({'success': True, 'message': 'If this email exists, a reset link has been sent.'})


@app.route('/auth/reset_password', methods=['POST'])
def auth_reset_password():
    body = request.json
    token = (body.get('token') or '').strip()
    new_password = body.get('password') or ''

    if not token:
        return jsonify({'error': 'Reset token is missing'}), 400

    pw_err = _validate_password(new_password)
    if pw_err:
        return jsonify({'error': pw_err}), 400

    entry = db_get_pending_reset(token)
    if not entry:
        return jsonify({'error': 'Invalid or already used reset link'}), 400
    if time.time() > entry['expires']:
        db_delete_pending_reset_by_token(token)
        return jsonify({'error': 'Reset link has expired. Please request a new one.'}), 400

    user = db_get_user(entry['email'])
    if not user:
        return jsonify({'error': 'User not found'}), 404

    db_update_user_fields(entry['email'], password_hash=generate_password_hash(new_password))
    db_delete_pending_reset_by_token(token)
    db_delete_sessions_by_email(entry['email'])  # Invalidate all sessions for security

    return jsonify({'success': True, 'message': 'Password changed successfully. You can now log in.'})


@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    db_delete_session(token)
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)