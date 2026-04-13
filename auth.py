import os
import json
import hashlib
import hmac
import base64
import time
from functools import wraps
from flask import request, jsonify

ROOT = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(ROOT, 'data', 'users.json')

SECRET_KEY = os.environ.get('SECRET_KEY', '').strip()
if not SECRET_KEY:
    # Stable fallback for local/dev runs so tokens don't invalidate on restart.
    SECRET_KEY = 'defectsense-dev-secret-change-me'

_LOGIN_ATTEMPTS: dict = {}
_MAX_ATTEMPTS = 10
_LOCKOUT_SECONDS = 300


def _pbkdf2_hash(password: str, salt: str = '') -> str:
    if not salt:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 260000)
    return f"pbkdf2$sha256$260000${salt}${dk.hex()}"


def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        if stored.startswith('pbkdf2$'):
            _, algo, iters, salt, dk_stored = stored.split('$')
            dk = hashlib.pbkdf2_hmac(algo, password.encode(), salt.encode(), int(iters))
            return hmac.compare_digest(dk.hex(), dk_stored)
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(legacy_hash, stored)
    except Exception:
        return False


def _default_users() -> dict:
    admin_pw = os.environ.get('ADMIN_PASSWORD', '')
    demo_pw = os.environ.get('DEMO_PASSWORD', '')
    professor_pw = os.environ.get('PROFESSOR_PASSWORD', '')

    if not admin_pw:
        admin_pw = 'admin123'
    if not demo_pw:
        demo_pw = 'demo123'
    if not professor_pw:
        professor_pw = 'seai2024'

    return {
        'admin': {
            'password_hash': _pbkdf2_hash(admin_pw),
            'role': 'admin',
            'name': 'Admin User',
            'created_at': '2024-01-01',
            'predictions_count': 0,
        },
        'demo': {
            'password_hash': _pbkdf2_hash(demo_pw),
            'role': 'analyst',
            'name': 'Demo Analyst',
            'created_at': '2024-01-01',
            'predictions_count': 0,
        },
        'professor': {
            'password_hash': _pbkdf2_hash(professor_pw),
            'role': 'viewer',
            'name': 'Prof. Evaluator',
            'created_at': '2024-01-01',
            'predictions_count': 0,
        },
    }


def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    defaults = _default_users()
    _save_users(defaults)
    return defaults


def _save_users(users: dict) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()


def _b64decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += '=' * padding
    return base64.urlsafe_b64decode(s)


def _is_rate_limited(username: str) -> bool:
    now = time.time()
    entry = _LOGIN_ATTEMPTS.get(username, {'count': 0, 'locked_until': 0})
    if now < entry.get('locked_until', 0):
        return True
    if now - entry.get('first_attempt', now) > _LOCKOUT_SECONDS:
        _LOGIN_ATTEMPTS.pop(username, None)
    return False


def _record_failed_attempt(username: str) -> None:
    now = time.time()
    entry = _LOGIN_ATTEMPTS.setdefault(username, {'count': 0, 'first_attempt': now, 'locked_until': 0})
    entry['count'] += 1
    if entry['count'] >= _MAX_ATTEMPTS:
        entry['locked_until'] = now + _LOCKOUT_SECONDS


def _clear_attempts(username: str) -> None:
    _LOGIN_ATTEMPTS.pop(username, None)


def create_token(username: str, role: str) -> str:
    header = _b64encode(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode())
    payload = _b64encode(json.dumps({
        'sub': username,
        'role': role,
        'iat': int(time.time()),
        'exp': int(time.time()) + 86400,
    }).encode())
    sig_input = f"{header}.{payload}"
    sig = hmac.new(SECRET_KEY.encode(), sig_input.encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64encode(sig)}"


def verify_token(token: str) -> dict | None:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        sig_input = f"{header}.{payload}"
        expected_sig = hmac.new(SECRET_KEY.encode(), sig_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, _b64encode(expected_sig)):
            return None
        data = json.loads(_b64decode(payload).decode())
        if data.get('exp', 0) < int(time.time()):
            return None
        return data
    except Exception:
        return None


def login(username: str, password: str) -> dict | None:
    if _is_rate_limited(username):
        return None

    users = _load_users()
    if username not in users:
        _record_failed_attempt(username)
        return None

    user = users[username]
    if not _pbkdf2_verify(password, user['password_hash']):
        _record_failed_attempt(username)
        return None

    if not user['password_hash'].startswith('pbkdf2$'):
        users[username]['password_hash'] = _pbkdf2_hash(password)
        _save_users(users)

    _clear_attempts(username)
    token = create_token(username, user['role'])
    return {
        'token': token,
        'username': username,
        'name': user.get('name', username),
        'role': user['role'],
    }


def register(username: str, password: str, name: str) -> dict | None:
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 8:
        return None

    users = _load_users()
    if username in users:
        return None

    users[username] = {
        'password_hash': _pbkdf2_hash(password),
        'role': 'analyst',
        'name': name.strip() or username,
        'created_at': time.strftime('%Y-%m-%d'),
        'predictions_count': 0,
    }
    _save_users(users)
    token = create_token(username, 'analyst')
    return {
        'token': token,
        'username': username,
        'name': name.strip() or username,
        'role': 'analyst',
    }


def get_user(username: str) -> dict | None:
    return _load_users().get(username)


def increment_prediction_count(username: str) -> None:
    users = _load_users()
    if username in users:
        users[username]['predictions_count'] = users[username].get('predictions_count', 0) + 1
        _save_users(users)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = None
        bearer_token = None
        if auth_header.startswith('Bearer '):
            bearer_token = auth_header[7:]

        if bearer_token:
            payload = verify_token(bearer_token)
            if payload:
                request.current_user = payload
                return f(*args, **kwargs)

        # Fallback to cookie token when bearer token is missing or stale.
        if 'token' in request.cookies:
            token = request.cookies.get('token')
        else:
            token = bearer_token

        if not token:
            return jsonify({'error': 'Authentication required', 'code': 'NO_TOKEN'}), 401

        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token', 'code': 'INVALID_TOKEN'}), 401

        request.current_user = payload
        return f(*args, **kwargs)
    return decorated