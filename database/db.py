"""
MySQL database module for FER System.
Automatically creates the database and all tables on first run.
Migrates existing JSON files to MySQL if they exist.
"""
import json
import os
import time

import mysql.connector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_PATH = os.path.join(BASE_DIR, 'database', 'db_config.json')


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg():
    if os.path.exists(_CFG_PATH):
        with open(_CFG_PATH) as f:
            return json.load(f)
    return {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',
        'database': 'fer_database'
    }


def get_conn():
    cfg = _load_cfg()
    return mysql.connector.connect(
        host=cfg['host'],
        port=int(cfg['port']),
        user=cfg['user'],
        password=cfg['password'],
        database=cfg['database'],
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci'
    )


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create the database and all tables if they don't exist, then migrate JSON data."""
    cfg = _load_cfg()

    # Connect without specifying the database so we can create it
    conn = mysql.connector.connect(
        host=cfg['host'],
        port=int(cfg['port']),
        user=cfg['user'],
        password=cfg['password'],
        charset='utf8mb4'
    )
    cur = conn.cursor()
    db_name = cfg['database']
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cur.execute(f"USE `{db_name}`")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email         VARCHAR(255) PRIMARY KEY,
            password_hash TEXT         NOT NULL,
            username      VARCHAR(100),
            avatar        LONGTEXT,
            verified      TINYINT(1)   DEFAULT 0,
            created_at    DOUBLE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token      VARCHAR(36) PRIMARY KEY,
            email      VARCHAR(255),
            created_at DOUBLE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_2fa (
            email   VARCHAR(255) PRIMARY KEY,
            code    VARCHAR(10),
            expires DOUBLE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_verification (
            email   VARCHAR(255) PRIMARY KEY,
            code    VARCHAR(10),
            expires DOUBLE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pending_reset (
            token   VARCHAR(36) PRIMARY KEY,
            email   VARCHAR(255),
            expires DOUBLE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)


    cur.execute("""
        CREATE TABLE IF NOT EXISTS archive (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            image_path  VARCHAR(500),
            timestamp   DOUBLE,
            user_email  VARCHAR(255),
            faces       JSON,
            custom_name VARCHAR(255),
            INDEX idx_image_path (image_path(255)),
            INDEX idx_user_email (user_email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    conn.commit()
    cur.close()
    conn.close()

    _migrate_from_json()
    print('[DB] Database ready.')


# ── Migration ─────────────────────────────────────────────────────────────────

def _migrate_from_json():
    """One-time migration from JSON files to MySQL. Renames JSON to .migrated after import."""

    # --- users.json ---
    users_json = os.path.join(BASE_DIR, 'database', 'users.json')
    if os.path.exists(users_json):
        try:
            with open(users_json) as f:
                data = json.load(f)
            conn = get_conn()
            cur = conn.cursor()

            for u in data.get('users', []):
                cur.execute(
                    """INSERT IGNORE INTO users
                       (email, password_hash, username, avatar, verified, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (u['email'], u['password_hash'], u.get('username', ''),
                     u.get('avatar', ''), int(bool(u.get('verified', False))),
                     u.get('created_at', time.time()))
                )

            for s in data.get('sessions', []):
                cur.execute(
                    "INSERT IGNORE INTO sessions (token, email, created_at) VALUES (%s, %s, %s)",
                    (s['token'], s['email'], s.get('created_at', time.time()))
                )

            for e in data.get('pending_2fa', []):
                cur.execute(
                    "INSERT IGNORE INTO pending_2fa (email, code, expires) VALUES (%s, %s, %s)",
                    (e['email'], e['code'], e['expires'])
                )

            for e in data.get('pending_verification', []):
                cur.execute(
                    "INSERT IGNORE INTO pending_verification (email, code, expires) VALUES (%s, %s, %s)",
                    (e['email'], e['code'], e['expires'])
                )

            for e in data.get('pending_reset', []):
                cur.execute(
                    "INSERT IGNORE INTO pending_reset (token, email, expires) VALUES (%s, %s, %s)",
                    (e['token'], e['email'], e['expires'])
                )

            conn.commit()
            cur.close()
            conn.close()
            os.rename(users_json, users_json + '.migrated')
            print('[DB] Migrated users.json → MySQL')
        except Exception as ex:
            print(f'[DB] Migration warning (users.json): {ex}')

    # --- archive.json ---
    archive_json = os.path.join(BASE_DIR, 'database', 'archive.json')
    if os.path.exists(archive_json):
        try:
            with open(archive_json) as f:
                entries = json.load(f)
            conn = get_conn()
            cur = conn.cursor()
            for e in entries:
                cur.execute(
                    """INSERT INTO archive (image_path, timestamp, user_email, faces, custom_name)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (e.get('image_path'), e.get('timestamp'),
                     e.get('user_email'), json.dumps(e.get('faces', {})),
                     e.get('custom_name'))
                )
            conn.commit()
            cur.close()
            conn.close()
            os.rename(archive_json, archive_json + '.migrated')
            print('[DB] Migrated archive.json → MySQL')
        except Exception as ex:
            print(f'[DB] Migration warning (archive.json): {ex}')


# ── User helpers ──────────────────────────────────────────────────────────────

def db_get_user(email):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def db_create_user(email, password_hash, username, avatar):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO users (email, password_hash, username, avatar, verified, created_at)
           VALUES (%s, %s, %s, %s, 0, %s)""",
        (email, password_hash, username, avatar, time.time())
    )
    conn.commit()
    cur.close()
    conn.close()


def db_update_user_fields(email, **fields):
    if not fields:
        return
    sets = ', '.join(f'`{k}`=%s' for k in fields)
    vals = list(fields.values()) + [email]
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE users SET {sets} WHERE email=%s", vals)
    conn.commit()
    cur.close()
    conn.close()


# ── Session helpers ───────────────────────────────────────────────────────────

def db_get_session_user(token):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT u.* FROM sessions s
        JOIN users u ON u.email = s.email
        WHERE s.token = %s
    """, (token,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def db_create_session(token, email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token, email, created_at) VALUES (%s, %s, %s)",
        (token, email, time.time())
    )
    conn.commit()
    cur.close()
    conn.close()


def db_delete_session(token):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token=%s", (token,))
    conn.commit()
    cur.close()
    conn.close()


def db_delete_sessions_by_email(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE email=%s", (email,))
    conn.commit()
    cur.close()
    conn.close()


# ── Pending 2FA ───────────────────────────────────────────────────────────────

def db_set_pending_2fa(email, code, expires):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO pending_2fa (email, code, expires) VALUES (%s, %s, %s)",
        (email, code, expires)
    )
    conn.commit()
    cur.close()
    conn.close()


def db_get_pending_2fa(email):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM pending_2fa WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def db_delete_pending_2fa(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM pending_2fa WHERE email=%s", (email,))
    conn.commit()
    cur.close()
    conn.close()


# ── Pending verification ──────────────────────────────────────────────────────

def db_set_pending_verification(email, code, expires):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO pending_verification (email, code, expires) VALUES (%s, %s, %s)",
        (email, code, expires)
    )
    conn.commit()
    cur.close()
    conn.close()


def db_get_pending_verification(email):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM pending_verification WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def db_delete_pending_verification(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM pending_verification WHERE email=%s", (email,))
    conn.commit()
    cur.close()
    conn.close()


# ── Pending reset ─────────────────────────────────────────────────────────────

def db_set_pending_reset(email, token, expires):
    conn = get_conn()
    cur = conn.cursor()
    # Invalidate any existing reset tokens for this user first
    cur.execute("DELETE FROM pending_reset WHERE email=%s", (email,))
    cur.execute(
        "INSERT INTO pending_reset (token, email, expires) VALUES (%s, %s, %s)",
        (token, email, expires)
    )
    conn.commit()
    cur.close()
    conn.close()


def db_get_pending_reset(token):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM pending_reset WHERE token=%s", (token,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def db_delete_pending_reset_by_token(token):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM pending_reset WHERE token=%s", (token,))
    conn.commit()
    cur.close()
    conn.close()


# ── Archive ───────────────────────────────────────────────────────────────────

def db_save_archive(image_path, timestamp, user_email, faces):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO archive (image_path, timestamp, user_email, faces) VALUES (%s, %s, %s, %s)",
        (image_path, timestamp, user_email, json.dumps(faces))
    )
    conn.commit()
    cur.close()
    conn.close()


def db_get_archive_by_user(email):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM archive WHERE user_email=%s ORDER BY timestamp DESC",
        (email,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    for r in rows:
        if isinstance(r.get('faces'), str):
            r['faces'] = json.loads(r['faces'])
    return rows


def db_get_archive_entry(image_path):
    """Return single archive entry by image_path (tries both /uploads/X and X)."""
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM archive WHERE image_path=%s LIMIT 1", (image_path,))
    row = cur.fetchone()
    if not row:
        # Try with /uploads/ prefix
        alt = image_path if image_path.startswith('/uploads/') else f'/uploads/{os.path.basename(image_path)}'
        cur.execute("SELECT * FROM archive WHERE image_path=%s LIMIT 1", (alt,))
        row = cur.fetchone()
    cur.close()
    conn.close()
    if row and isinstance(row.get('faces'), str):
        row['faces'] = json.loads(row['faces'])
    return row


def db_delete_archive_entry(image_path):
    """Delete archive entry by image_path. Returns True if a row was deleted."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM archive WHERE image_path=%s", (image_path,))
    affected = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return affected > 0


def db_update_archive_entry(image_path, faces=None, custom_name=None):
    """Update faces and/or custom_name for an archive entry. Returns True if found."""
    if faces is None and custom_name is None:
        return False
    conn = get_conn()
    cur = conn.cursor()
    if faces is not None and custom_name is not None:
        cur.execute(
            "UPDATE archive SET faces=%s, custom_name=%s WHERE image_path=%s",
            (json.dumps(faces), custom_name, image_path)
        )
    elif faces is not None:
        cur.execute(
            "UPDATE archive SET faces=%s WHERE image_path=%s",
            (json.dumps(faces), image_path)
        )
    else:
        cur.execute(
            "UPDATE archive SET custom_name=%s WHERE image_path=%s",
            (custom_name, image_path)
        )
    affected = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return affected > 0


def db_update_archive_image_path(old_path, new_path, faces=None):
    """Rename image_path and optionally update faces in archive."""
    conn = get_conn()
    cur = conn.cursor()
    if faces is not None:
        cur.execute(
            "UPDATE archive SET image_path=%s, faces=%s WHERE image_path=%s",
            (new_path, json.dumps(faces), old_path)
        )
    else:
        cur.execute(
            "UPDATE archive SET image_path=%s WHERE image_path=%s",
            (new_path, old_path)
        )
    conn.commit()
    cur.close()
    conn.close()
