import hashlib
import json
import http.client
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_HISTORY_FILE = ROOT / 'data' / 'pipeline_history.json'
K8S_DIR = ROOT / 'k8s'
DOCKER_SOCK = Path('/var/run/docker.sock')


class _UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__('localhost')
        self._socket_path = socket_path

    def connect(self):
        import socket
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


def _run_command(command, cwd=None, timeout=900):
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except FileNotFoundError:
        return 127, '', f'Executable not found: {command[0]}'
    except subprocess.TimeoutExpired:
        return 1, '', f'Command timed out after {timeout}s: {" ".join(command)}'
    except Exception as exc:
        return 1, '', str(exc)


def _stage_result(stage_name, command, cwd=None, timeout=900):
    started_at = time.time()
    returncode, stdout, stderr = _run_command(command, cwd=cwd, timeout=timeout)
    return {
        'stage': stage_name,
        'command': ' '.join(command),
        'returncode': returncode,
        'stdout': stdout,
        'stderr': stderr,
        'duration': time.time() - started_at,
    }


def _load_dockerignore_patterns(context_dir: Path) -> set:
    dockerignore = context_dir / '.dockerignore'
    patterns = {'.git', '.venv', '__pycache__', '*.pyc', '.env', 'node_modules'}
    if dockerignore.exists():
        for line in dockerignore.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                patterns.add(line)
    return patterns


def _should_ignore(rel_str: str, patterns: set) -> bool:
    parts = Path(rel_str).parts
    for pattern in patterns:
        clean = pattern.lstrip('/')
        if rel_str.startswith(clean):
            return True
        if parts and parts[0] == clean:
            return True
        if Path(rel_str).name == clean:
            return True
        if '*' in clean:
            import fnmatch
            if fnmatch.fnmatch(rel_str, clean) or fnmatch.fnmatch(Path(rel_str).name, clean):
                return True
    return False


def _docker_build_via_socket(image_tag: str, context_dir: Path) -> dict:
    started_at = time.time()

    if not DOCKER_SOCK.exists():
        return {
            'stage': 'build',
            'command': 'docker-engine-api build',
            'returncode': 127,
            'stdout': '',
            'stderr': f'Docker socket not found at {DOCKER_SOCK}',
            'duration': time.time() - started_at,
        }

    tar_path = None
    conn = None
    try:
        ignore_patterns = _load_dockerignore_patterns(context_dir)

        with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as tmp:
            tar_path = tmp.name

        with tarfile.open(tar_path, 'w') as tar:
            for path in context_dir.rglob('*'):
                rel = path.relative_to(context_dir)
                rel_str = str(rel)
                if _should_ignore(rel_str, ignore_patterns):
                    continue
                tar.add(path, arcname=rel_str)

        with open(tar_path, 'rb') as f:
            body = f.read()

        conn = _UnixSocketHTTPConnection(str(DOCKER_SOCK))
        endpoint = f"/v1.41/build?t={quote(image_tag)}&rm=1"
        conn.request(
            'POST',
            endpoint,
            body=body,
            headers={
                'Content-Type': 'application/x-tar',
                'Content-Length': str(len(body)),
            },
        )
        response = conn.getresponse()
        raw = response.read().decode('utf-8', errors='ignore')
        error_lines = [line for line in raw.splitlines() if 'errorDetail' in line or '"error"' in line]

        if response.status >= 400 or error_lines:
            return {
                'stage': 'build',
                'command': 'docker-engine-api build',
                'returncode': 1,
                'stdout': raw[-1200:],
                'stderr': error_lines[-1] if error_lines else f'Docker API HTTP {response.status}',
                'duration': time.time() - started_at,
            }

        return {
            'stage': 'build',
            'command': 'docker-engine-api build',
            'returncode': 0,
            'stdout': raw[-1200:],
            'stderr': '',
            'duration': time.time() - started_at,
        }
    except Exception as exc:
        return {
            'stage': 'build',
            'command': 'docker-engine-api build',
            'returncode': 1,
            'stdout': '',
            'stderr': str(exc),
            'duration': time.time() - started_at,
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if tar_path and os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except Exception:
                pass


def _is_k8s_reachable() -> tuple[bool, str]:
    returncode, stdout, stderr = _run_command(['kubectl', 'cluster-info'], cwd=ROOT, timeout=15)
    if returncode == 0:
        return True, ''
    return False, stderr or stdout or 'Kubernetes cluster is not reachable.'


def _is_kubectl_available() -> bool:
    return shutil.which('kubectl') is not None


def _validate_k8s_manifests_local(k8s_dir: Path) -> tuple[bool, str]:
    try:
        import yaml
    except Exception as exc:
        return False, f'PyYAML unavailable for local validation: {exc}'

    if not k8s_dir.exists():
        return False, 'k8s/ manifests directory is missing'

    yaml_files = sorted(list(k8s_dir.glob('*.yml')) + list(k8s_dir.glob('*.yaml')))
    if not yaml_files:
        return False, 'No Kubernetes manifest files found in k8s/'

    checked = 0
    for path in yaml_files:
        content = path.read_text(encoding='utf-8', errors='ignore')
        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue
            checked += 1
            if not isinstance(doc, dict):
                return False, f'Invalid YAML doc in {path.name}: expected mapping'
            if 'apiVersion' not in doc or 'kind' not in doc:
                return False, f'Missing apiVersion/kind in {path.name}'

    return True, f'Validated {checked} Kubernetes manifest document(s) locally'


def _compile_source_tempfile(source_code: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix='defectsense-pipeline-')
    temp_path = Path(temp_dir) / 'uploaded_file.py'
    temp_path.write_text(source_code, encoding='utf-8')
    return str(temp_path)


def _analyze_security(source_code: str) -> dict:
    issues = []
    lowered = source_code.lower()

    if 'eval(' in lowered:
        issues.append({'severity': 'high', 'rule': 'B307', 'message': 'Use of eval() detected'})
    if 'exec(' in lowered:
        issues.append({'severity': 'high', 'rule': 'B102', 'message': 'Use of exec() detected'})
    if 'subprocess' in lowered and 'shell=true' in lowered:
        issues.append({'severity': 'medium', 'rule': 'B602', 'message': 'subprocess call with shell=True detected'})
    if 'pickle.load' in lowered:
        issues.append({'severity': 'medium', 'rule': 'B301', 'message': 'Pickle deserialization detected'})
    if 'os.system(' in lowered:
        issues.append({'severity': 'medium', 'rule': 'B605', 'message': 'os.system() call detected — use subprocess instead'})
    if 'hashlib.md5(' in lowered:
        issues.append({'severity': 'low', 'rule': 'B303', 'message': 'MD5 is not collision-resistant — use SHA-256 for security'})

    bandit_result = None
    if shutil.which('bandit'):
        bandit_result = _stage_result('bandit', ['bandit', '-q', '-r', str(ROOT), '-f', 'json'], cwd=ROOT, timeout=900)
        if bandit_result['returncode'] == 0 and bandit_result['stdout']:
            try:
                payload = json.loads(bandit_result['stdout'])
                for item in payload.get('results', []):
                    issues.append({
                        'severity': item.get('issue_severity', 'medium').lower(),
                        'rule': item.get('test_id', 'B000'),
                        'message': item.get('issue_text', 'Security issue detected'),
                    })
            except Exception:
                pass

    if not issues:
        issues.append({'severity': 'pass', 'rule': 'OK', 'message': 'No security vulnerabilities found'})

    pip_check = _stage_result('pip-check', [sys.executable, '-m', 'pip', 'check'], cwd=ROOT, timeout=300)

    return {
        'issues': issues,
        'pip_check': pip_check,
        'bandit': bandit_result,
    }


def _emit_skipped_stage(stage_name: str, reason: str) -> dict:
    return {
        'stage': stage_name,
        'status': 'warning',
        'stdout': '',
        'stderr': reason,
        'returncode': 1,
        'duration': 0,
        'skipped': True,
    }


def run_pipeline_stream(filename: str, source_code: str, metrics: dict, prediction: dict):
    pipeline_id = hashlib.sha256(f"{filename}{time.time()}".encode()).hexdigest()[:8].upper()
    results = []
    final_status = 'success'
    blocked_reasons = []
    temp_source = None
    build_image = f"proj3-defectsense:{pipeline_id.lower()}"

    try:
        temp_source = _compile_source_tempfile(source_code)

        yield json.dumps({
            'type': 'pipeline_start',
            'pipeline_id': pipeline_id,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'mode': 'real',
        })

        yield json.dumps({'type': 'stage_start', 'stage': 'lint', 'name': 'Syntax Check', 'tool': 'python -m py_compile'})
        lint_result = _stage_result('lint', [sys.executable, '-m', 'py_compile', temp_source], cwd=ROOT, timeout=120)
        lint_result['status'] = 'pass' if lint_result['returncode'] == 0 else 'fail'
        results.append(lint_result)
        yield json.dumps({'type': 'stage_complete', 'stage': 'lint', 'status': lint_result['status'], 'duration': f"{lint_result['duration']:.1f}s", 'output': _summarize_result(lint_result)})
        lint_failed = lint_result['status'] == 'fail'
        if lint_failed:
            blocked_reasons.append('Syntax check failed')
            final_status = 'blocked'

        yield json.dumps({'type': 'stage_start', 'stage': 'unit_test', 'name': 'Unit Tests', 'tool': 'python -m unittest discover -v'})
        test_result = _stage_result('unit_test', [sys.executable, '-m', 'unittest', 'discover', '-v'], cwd=ROOT, timeout=900)
        test_text = f"{test_result.get('stdout', '')}\n{test_result.get('stderr', '')}"
        if test_result['returncode'] == 0:
            test_result['status'] = 'pass'
        elif 'Ran 0 tests' in test_text or 'NO TESTS RAN' in test_text:
            test_result['status'] = 'warning'
        else:
            test_result['status'] = 'fail'
            blocked_reasons.append('Unit tests failed')
            final_status = 'blocked'
        results.append(test_result)
        yield json.dumps({'type': 'stage_complete', 'stage': 'unit_test', 'status': test_result['status'], 'duration': f"{test_result['duration']:.1f}s", 'output': _summarize_result(test_result)})

        yield json.dumps({'type': 'stage_start', 'stage': 'defect_scan', 'name': 'ML Defect Gate', 'tool': 'stacking ensemble model'})
        defect_probability = float(prediction.get('probability', 0) or 0)
        risk_score = prediction.get('risk_score', {})
        defect_result = {
            'stage': 'defect_scan',
            'status': 'pass',
            'probability': defect_probability,
            'label': prediction.get('label'),
            'risk_score': risk_score,
            'top_risks': [f[0] for f in prediction.get('top_features', [])[:3]],
            'model_meta': prediction.get('model_meta', {}),
        }
        ml_gate_blocked = defect_probability > 0.70 and not lint_failed
        if ml_gate_blocked:
            blocked_reasons.append(f"ML defect gate triggered ({defect_probability * 100:.1f}% > 70%)")
            final_status = 'blocked'
        results.append(defect_result)
        yield json.dumps({'type': 'stage_complete', 'stage': 'defect_scan', 'status': 'pass', 'duration': '0.0s', 'output': defect_result})

        yield json.dumps({'type': 'stage_start', 'stage': 'security', 'name': 'Security Scan', 'tool': 'static rules + pip check + bandit'})
        security_result = _analyze_security(source_code)
        bandit_failed = (
            security_result['bandit'] is not None and
            security_result['bandit'].get('returncode') not in (0, None)
        )
        high_severity = any(i['severity'] == 'high' for i in security_result['issues'])
        if security_result['pip_check']['returncode'] != 0 or bandit_failed or high_severity:
            security_status = 'warning'
        elif security_result['issues'] and security_result['issues'][0]['severity'] == 'pass':
            security_status = 'pass'
        else:
            security_status = 'warning'
        security_result['status'] = security_status
        results.append(security_result)
        security_duration = float(security_result.get('pip_check', {}).get('duration', 0) or 0)
        yield json.dumps({'type': 'stage_complete', 'stage': 'security', 'status': security_status, 'duration': f"{security_duration:.1f}s", 'output': _summarize_result(security_result)})

        if lint_failed or ml_gate_blocked:
            yield json.dumps({'type': 'pipeline_blocked', 'reason': blocked_reasons[-1], 'stage': 'lint' if lint_failed else 'defect_scan'})
            for skipped_stage in ['build', 'k8s_deploy']:
                skip = _emit_skipped_stage(skipped_stage, blocked_reasons[-1])
                results.append(skip)
                yield json.dumps({'type': 'stage_complete', 'stage': skipped_stage, 'status': 'warning', 'duration': '0.0s', 'output': _summarize_result(skip)})

        else:
            yield json.dumps({'type': 'stage_start', 'stage': 'build', 'name': 'Docker Build', 'tool': 'docker build'})
            build_result = _stage_result('build', ['docker', 'build', '-t', build_image, '.'], cwd=ROOT, timeout=3600)
            build_result['image'] = build_image
            docker_cli_missing = build_result['returncode'] == 127 and 'Executable not found: docker' in (build_result.get('stderr') or '')
            used_socket_fallback = False

            if docker_cli_missing:
                used_socket_fallback = True
                build_result = _docker_build_via_socket(build_image, ROOT)
                build_result['image'] = build_image

            if build_result['returncode'] == 0:
                build_result['status'] = 'pass'
            elif docker_cli_missing and used_socket_fallback:
                build_result['status'] = 'warning'
                # Socket fallback was attempted; keep original engine error for diagnosis.
                build_result['stderr'] = build_result.get('stderr') or 'Docker Engine API build failed during socket fallback.'
                build_result['skipped'] = False
            else:
                build_result['status'] = 'fail'
                blocked_reasons.append('Docker build failed')
                final_status = 'blocked'

            results.append(build_result)
            yield json.dumps({'type': 'stage_complete', 'stage': 'build', 'status': build_result['status'], 'duration': f"{build_result['duration']:.1f}s", 'output': _summarize_result(build_result)})

            if build_result['status'] == 'pass' and K8S_DIR.exists():
                if not _is_kubectl_available():
                    ok, message = _validate_k8s_manifests_local(K8S_DIR)
                    deploy_result = {
                        'stage': 'k8s_deploy',
                        'status': 'pass' if ok else 'warning',
                        'stdout': message if ok else '',
                        'stderr': '' if ok else message,
                        'returncode': 0 if ok else 1,
                        'duration': 0,
                        'validated_only': True,
                    }
                    results.append(deploy_result)
                    yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': deploy_result['status'], 'duration': '0.0s', 'output': _summarize_result(deploy_result)})
                else:
                    reachable, reason = _is_k8s_reachable()
                    if not reachable:
                        dry_run = _stage_result('k8s_deploy', ['kubectl', 'apply', '--dry-run=client', '-f', str(K8S_DIR)], cwd=ROOT, timeout=300)
                        deploy_result = {
                            'stage': 'k8s_deploy',
                            'status': 'pass' if dry_run['returncode'] == 0 else 'warning',
                            'stdout': dry_run.get('stdout', ''),
                            'stderr': f"Cluster unreachable — client-side dry-run only. {reason}",
                            'returncode': dry_run['returncode'],
                            'duration': dry_run.get('duration', 0),
                            'validated_only': True,
                        }
                        results.append(deploy_result)
                        yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': deploy_result['status'], 'duration': f"{deploy_result['duration']:.1f}s", 'output': _summarize_result(deploy_result)})
                    else:
                        yield json.dumps({'type': 'stage_start', 'stage': 'k8s_deploy', 'name': 'Kubernetes Deploy', 'tool': 'kubectl apply'})
                        deploy_result = _stage_result('k8s_deploy', ['kubectl', 'apply', '-f', str(K8S_DIR)], cwd=ROOT, timeout=900)
                        rollout_result = _stage_result('k8s_rollout', ['kubectl', 'rollout', 'status', 'deployment/proj3-defectsense', '--timeout=180s'], cwd=ROOT, timeout=300)
                        deploy_result['rollout'] = rollout_result
                        if deploy_result['returncode'] == 0 and rollout_result['returncode'] == 0:
                            deploy_result['status'] = 'pass'
                        else:
                            deploy_result['status'] = 'fail'
                            blocked_reasons.append('Kubernetes deploy failed')
                            final_status = 'blocked'
                        results.append(deploy_result)
                        yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': deploy_result['status'], 'duration': f"{deploy_result['duration']:.1f}s", 'output': _summarize_result(deploy_result)})

            elif build_result.get('skipped'):
                skip = _emit_skipped_stage('k8s_deploy', 'Docker build was skipped in this runtime')
                results.append(skip)
                yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': 'warning', 'duration': '0.0s', 'output': _summarize_result(skip)})

            elif not K8S_DIR.exists():
                skip = _emit_skipped_stage('k8s_deploy', 'k8s/ manifests directory is missing')
                results.append(skip)
                yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': 'warning', 'duration': '0.0s', 'output': _summarize_result(skip)})

            else:
                skip = _emit_skipped_stage('k8s_deploy', 'Docker build failed')
                results.append(skip)
                yield json.dumps({'type': 'stage_complete', 'stage': 'k8s_deploy', 'status': 'warning', 'duration': '0.0s', 'output': _summarize_result(skip)})

        if final_status != 'blocked':
            warning_stages = [s.get('stage') for s in results if isinstance(s, dict) and s.get('status') == 'warning']
            if warning_stages:
                final_status = 'warning'

        if final_status == 'blocked' and blocked_reasons:
            yield json.dumps({'type': 'pipeline_blocked', 'reason': '; '.join(dict.fromkeys(blocked_reasons)), 'stage': 'pipeline'})

    finally:
        if temp_source and os.path.exists(temp_source):
            try:
                os.remove(temp_source)
                os.rmdir(os.path.dirname(temp_source))
            except Exception:
                pass

    _save_pipeline_run(pipeline_id, filename, prediction.get('probability', 0), final_status, results)

    warning_stages = [s.get('stage') for s in results if isinstance(s, dict) and s.get('status') == 'warning']
    warning_reason = ('Warnings in stages: ' + ', '.join(dict.fromkeys(warning_stages))) if final_status == 'warning' and warning_stages else ''

    total_duration = sum(float(s.get('duration', 0) or 0) for s in results if isinstance(s, dict))
    yield json.dumps({
        'type': 'pipeline_complete',
        'pipeline_id': pipeline_id,
        'status': final_status,
        'duration': f"{total_duration:.1f}s",
        'reason': '; '.join(dict.fromkeys(blocked_reasons)) if blocked_reasons else warning_reason,
        'mode': 'real',
    })


def _summarize_result(result: dict) -> dict:
    return {
        'command': result.get('command'),
        'returncode': result.get('returncode'),
        'stdout': result.get('stdout', '')[-1200:],
        'stderr': result.get('stderr', '')[-1200:],
        'status': result.get('status'),
        'duration': round(float(result.get('duration', 0) or 0), 2),
    }


def _save_pipeline_run(pipeline_id, filename, prob, status, results):
    try:
        os.makedirs(PIPELINE_HISTORY_FILE.parent, exist_ok=True)
        history = []
        if PIPELINE_HISTORY_FILE.exists():
            with open(PIPELINE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        history.insert(0, {
            'id': pipeline_id,
            'filename': filename,
            'probability': prob,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'stages': len(results),
            'mode': 'real',
        })
        with open(PIPELINE_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history[:20], f, indent=2)
    except Exception:
        pass


def get_pipeline_history() -> list:
    try:
        if PIPELINE_HISTORY_FILE.exists():
            with open(PIPELINE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []