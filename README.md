# Face Emotion Recognition System

> A web application for detecting and analyzing facial emotions in images and live camera feed - with AI-powered face enhancement and a full user account system.

---

## Screenshots

<img width="1864" height="938" alt="Image" src="https://github.com/user-attachments/assets/6613c83c-6696-4a20-9fba-7242fd7c1361" />
<img width="1864" height="938" alt="Image" src="https://github.com/user-attachments/assets/a6039d80-bcb9-48c0-ae22-1b1bd188e3f6" />
<img width="1860" height="942" alt="Image" src="https://github.com/user-attachments/assets/194e77fa-330c-482d-8dfe-3a10eb3ff032" />
<img width="1864" height="938" alt="Image" src="https://github.com/user-attachments/assets/f363cb9d-cce8-4175-97b7-c7027f055450" />
<img width="1864" height="938" alt="Image" src="https://github.com/user-attachments/assets/70aac83f-9cc2-4495-b7a9-260b842568c5" />
---

## What does it do?

Upload a photo or turn on your webcam - the app detects all faces and tells you what emotion each person is expressing. Before analysis, you can optionally enhance face quality using AI models (GFPGAN or Real-ESRGAN), which makes a big difference on low-resolution or older photos.

Results can be saved to a personal archive, browsed later, and managed from your user account.

---

## Features

- **Face detection** - OpenCV automatically highlights all faces in the image
- **Emotion analysis** - 7 emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`; scores are averaged over 10 passes with random augmentations for better accuracy
- **AI face enhancement** - two models to choose from:
  - **GFPGAN v1.4** - face restoration and reconstruction (great for blurry or old photos)
  - **Real-ESRGAN x4+** - super-resolution ×2 applied to each detected face crop
- **Live camera** - MJPEG stream with real-time emotion overlay; capture frames and save them to the archive
- **Analysis archive** - save, browse, rename, and delete analysis results tied to your account
- **User accounts**:
  - Registration with email verification
  - Login with 2FA code sent to email
  - Password reset via email link
  - Profile editing (username, avatar)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11.9, Flask |
| Database | MySQL 8.0 |
| Email | Flask-Mail (Gmail SMTP) |
| Face detection | OpenCV Haar Cascades |
| Emotion analysis | FER library (MTCNN backbone) |
| Face enhancement | GFPGAN, Real-ESRGAN, basicsr, PyTorch |
| Image processing | OpenCV, NumPy |

---

## Project Structure

```
Face-Emotion-Recognition/
├── FER.py                        # Flask server - all routes and business logic
├── templates/
│   └── combined.html             # Entire frontend as a single HTML file
├── database/
│   ├── db.py                     # MySQL helpers + automatic JSON migration
│   ├── db_config.json            # MySQL connection settings  ← do not commit
│   └── email_config.json         # SMTP credentials          ← do not commit
├── models/
│   ├── GFPGANv1.4.pth            # GFPGAN weights  (download manually, not in repo)
│   └── RealESRGAN_x4plus.pth     # Real-ESRGAN weights (download manually, not in repo)
├── uploads/                      # Uploaded and processed images (auto-created)
└── svgs/                         # UI icon assets
```

---

## Setup

### Requirements

- Python 3.11+
- MySQL 8.0+
- CUDA GPU (optional, but recommended for GFPGAN / Real-ESRGAN)

### 1. Install dependencies

```bash
pip install flask flask-mail opencv-python fer[tensorflow] mysql-connector-python \
            werkzeug numpy gfpgan realesrgan basicsr torch torchvision
```

> `fer` works with either TensorFlow or PyTorch - install whichever you already have.

### 2. Configure the database

Edit `database/db_config.json`:

```json
{
  "host": "localhost",
  "port": 3306,
  "user": "root",
  "password": "your_password",
  "database": "fer_database"
}
```

The database and all tables are **created automatically** on first run - no manual SQL needed.

### 3. Configure email

Edit `database/email_config.json`:

```json
{
  "smtp_host": "smtp.gmail.com",
  "smtp_port": 587,
  "smtp_ssl": false,
  "smtp_username": "your_address@gmail.com",
  "smtp_password": "your_app_password",
  "from_address": "your_address@gmail.com",
  "from_name": "FER System",
  "base_url": "http://localhost:5000"
}
```

> For Gmail you need an **App Password**, not your regular account password. Generate one at: Google Account → Security → 2-Step Verification → App passwords. [Instructions here.](https://support.google.com/accounts/answer/185833)

### 4. Download model weights

Place both files inside the `models/` directory:

| File | Download |
|---|---|
| `GFPGANv1.4.pth` | [GFPGAN Releases](https://github.com/TencentARC/GFPGAN/releases) |
| `RealESRGAN_x4plus.pth` | [Real-ESRGAN Releases](https://github.com/xinntao/Real-ESRGAN/releases) |

> Both files are excluded from version control via `.gitignore` - they can be hundreds of MB in size.

### 5. Run

```bash
python FER.py
```

Open your browser at `http://localhost:5000`.

---

## API Reference

<details>
<summary><strong>Image Analysis</strong></summary>

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_image` | Upload an image; returns detected face bounding boxes |
| `POST` | `/analyze_faces` | Analyze emotions on selected faces; returns scores + annotated image |
| `POST` | `/enhance_faces` | Enhance face crops with GFPGAN or Real-ESRGAN |
| `POST` | `/update_image_emotions` | Re-run analysis on an already-uploaded image |

</details>

<details>
<summary><strong>Camera</strong></summary>

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/start_camera` | Start the webcam capture thread |
| `POST` | `/stop_camera` | Stop the webcam |
| `GET` | `/camera_stream` | MJPEG live stream |
| `POST` | `/capture_frame` | Capture the current frame and analyze emotions |

</details>

<details>
<summary><strong>Archive</strong></summary>

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/save_results` | Save an analysis result to the archive |
| `GET` | `/get_archive` | List all archive entries for the logged-in user |
| `POST` | `/load_archive_image` | Load a specific archive entry |
| `POST` | `/delete_archive_entry` | Delete an archive entry and its image file |
| `POST` | `/update_archive_entry` | Rename or update an archive entry |

</details>

<details>
<summary><strong>Authentication</strong></summary>

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Register - sends email verification code |
| `POST` | `/auth/verify_registration` | Submit the email verification code |
| `POST` | `/auth/resend_verification` | Resend the verification code |
| `POST` | `/auth/login` | Log in - sends a 2FA code to email |
| `POST` | `/auth/verify_2fa` | Submit the 2FA code; creates a session cookie |
| `GET` | `/auth/me` | Get current user data |
| `POST` | `/auth/update_profile` | Update username and/or avatar |
| `POST` | `/auth/forgot_password` | Send a password reset link |
| `POST` | `/auth/reset_password` | Set new password using the reset token |
| `POST` | `/auth/logout` | Destroy the current session |

</details>

---

## Database Schema

```
users                - user accounts (email, password hash, username, avatar, verified flag)
sessions             - active sessions (UUID token → email)
pending_2fa          - pending 2FA codes with expiry
pending_verification - pending registration verification codes
pending_reset        - pending password reset tokens
archive              - saved analyses (file path, timestamp, user email, emotions JSON, name)
```

On first run the app automatically migrates any existing `users.json` / `archive.json` files to MySQL and renames them to `*.migrated`.

---

## Notes

- AI models (GFPGAN, Real-ESRGAN) are **lazy-loaded** on first use - startup is fast regardless of whether the models are downloaded.
- The `uploads/` folder grows over time - processed images are not deleted automatically. Clean it up manually or remove entries through the archive UI.
- Maximum upload size: **16 MB**.
- This app is designed for local / intranet use. For a public deployment, put it behind a proper WSGI server (e.g. Gunicorn) and use HTTPS.
- **Password reset emails may be blocked by spam filters.** If the reset link never arrives in your inbox, check the terminal where `FER.py` is running - the full reset URL is always printed there as a fallback:
  ```
  [RESET LINK] http://localhost:5000/reset?token=...
  ```

