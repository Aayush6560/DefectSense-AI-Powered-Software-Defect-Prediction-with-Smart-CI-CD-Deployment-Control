import os
import re
from flask import Blueprint, request, jsonify, make_response
from auth import login, register, verify_token, _is_rate_limited

auth_bp = Blueprint('auth', __name__)

_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_\-]{3,32}$')


def _set_auth_cookie(response, token: str):
    secure_cookie = os.environ.get('COOKIE_SECURE', 'false').lower() == 'true'
    response.set_cookie(
        'token',
        token,
        httponly=True,
        samesite='Lax',
        secure=secure_cookie,
        max_age=86400,
        path='/',
    )
    return response


@auth_bp.route('/api/auth/login', methods=['POST'])
def do_login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    if _is_rate_limited(username):
        return jsonify({'error': 'Too many failed attempts. Try again in 5 minutes.', 'code': 'RATE_LIMITED'}), 429

    result = login(username, password)
    if not result:
        return jsonify({'error': 'Invalid credentials'}), 401
    response = make_response(jsonify(result), 200)
    return _set_auth_cookie(response, result['token'])


@auth_bp.route('/api/auth/register', methods=['POST'])
def do_register():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''
    name = (data.get('name') or '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    if not _USERNAME_RE.match(username):
        return jsonify({'error': 'Username must be 3-32 characters: letters, numbers, _ or -'}), 400

    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    if len(name) > 64:
        return jsonify({'error': 'Name too long (max 64 characters)'}), 400

    result = register(username, password, name)
    if not result:
        return jsonify({'error': 'Username already taken'}), 409
    response = make_response(jsonify(result), 201)
    return _set_auth_cookie(response, result['token'])


@auth_bp.route('/api/auth/logout', methods=['POST'])
def do_logout():
    response = make_response(jsonify({'ok': True}), 200)
    response.delete_cookie('token', path='/')
    return response


@auth_bp.route('/api/auth/verify', methods=['GET'])
def verify():
    token = None
    bearer_token = None
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        bearer_token = auth_header[7:]

    if bearer_token:
        payload = verify_token(bearer_token)
        if payload:
            return jsonify({'valid': True, 'user': payload}), 200

    if 'token' in request.cookies:
        token = request.cookies.get('token')
    else:
        token = bearer_token

    if not token:
        return jsonify({'valid': False, 'code': 'NO_TOKEN'}), 401

    payload = verify_token(token)
    if not payload:
        return jsonify({'valid': False, 'code': 'INVALID_TOKEN'}), 401

    return jsonify({'valid': True, 'user': payload}), 200