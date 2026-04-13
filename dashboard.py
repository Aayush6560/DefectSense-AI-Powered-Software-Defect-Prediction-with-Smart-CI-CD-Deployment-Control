import os
import json
import subprocess
import psutil
import time
from flask import Blueprint, jsonify, render_template
from auth import require_auth
from ml.predict import get_model_meta, is_model_loaded
from pipeline import get_pipeline_history

dashboard_bp = Blueprint('dashboard', __name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_HISTORY_FILE = os.path.join(ROOT, 'data', 'metrics_history.json')
_METRICS_MAX_POINTS = 60


def _run_cmd(cmd, timeout=3):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception:
        return None, '', ''


def _get_docker_containers():
    returncode, output, stderr = _run_cmd(
        ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}']
    )
    if returncode is None:
        return [], 'Docker CLI is not available on this machine.', False
    if returncode != 0:
        return [], stderr or 'Docker daemon is not running.', False

    containers = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        containers.append({
            'name': parts[0] if len(parts) > 0 else 'unknown',
            'status': parts[1] if len(parts) > 1 else 'unknown',
            'image': parts[2] if len(parts) > 2 else 'unknown',
            'state': 'running' if 'Up' in (parts[1] if len(parts) > 1 else '') else 'stopped',
        })
    return containers, ('' if containers else 'No running Docker containers detected.'), True


def _get_k8s_pods():
    returncode, output, stderr = _run_cmd([
        'kubectl', 'get', 'pods', '--all-namespaces',
        '--no-headers', '-o',
        'custom-columns=NS:.metadata.namespace,NAME:.metadata.name,READY:.status.containerStatuses[0].ready,STATUS:.status.phase',
    ])
    if returncode is None:
        return [], 'kubectl is not available on this machine.', False
    if returncode != 0:
        return [], stderr or 'Kubernetes cluster is not reachable.', False

    pods = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            pods.append({
                'namespace': parts[0],
                'name': parts[1],
                'ready': parts[2],
                'status': parts[3],
            })
    return pods, ('' if pods else 'No Kubernetes pods detected.'), True


def _record_metrics_history(cpu: float, mem: float) -> None:
    try:
        os.makedirs(os.path.dirname(METRICS_HISTORY_FILE), exist_ok=True)
        history = []
        if os.path.exists(METRICS_HISTORY_FILE):
            with open(METRICS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        history.append({'ts': int(time.time()), 'cpu': round(cpu, 1), 'mem': round(mem, 1)})
        history = history[-_METRICS_MAX_POINTS:]
        with open(METRICS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f)
    except Exception:
        pass


@dashboard_bp.route('/dashboard')
@require_auth
def dashboard():
    return render_template('dashboard.html')


@dashboard_bp.route('/api/system-status')
@require_auth
def system_status():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()

    _record_metrics_history(cpu, mem.percent)

    containers, docker_note, docker_available = _get_docker_containers()
    pods, k8s_note, k8s_available = _get_k8s_pods()
    model_meta = get_model_meta()
    pipeline_history = get_pipeline_history()

    return jsonify({
        'cpu_percent': round(cpu, 1),
        'memory': {
            'percent': round(mem.percent, 1),
            'used_gb': round(mem.used / 1e9, 2),
            'total_gb': round(mem.total / 1e9, 2),
        },
        'disk': {
            'percent': round(disk.percent, 1),
            'used_gb': round(disk.used / 1e9, 1),
            'total_gb': round(disk.total / 1e9, 1),
        },
        'network': {
            'bytes_sent_mb': round(net.bytes_sent / 1e6, 1),
            'bytes_recv_mb': round(net.bytes_recv / 1e6, 1),
        },
        'containers': containers,
        'docker': {'available': docker_available, 'message': docker_note},
        'pods': pods,
        'k8s': {'available': k8s_available, 'message': k8s_note},
        'model': {
            'loaded': is_model_loaded(),
            'selected_model': model_meta.get('selected_model', 'Stacking Ensemble'),
            'architecture': model_meta.get('architecture', 'RF+GBT+ET+DT → LR'),
            'auc_roc': model_meta.get('auc_roc', 0.0),
            'f1_at_threshold': model_meta.get('f1_at_threshold', 0.0),
            'decision_threshold': model_meta.get('decision_threshold', 0.5),
            'dataset': model_meta.get('dataset', 'NASA KC1 PROMISE Repository'),
            'features': 21,
            'n_train': model_meta.get('n_train', 0),
            'smote_applied': model_meta.get('smote_applied', False),
        },
        'pipeline_runs': len(pipeline_history),
        'recent_pipelines': pipeline_history[:5],
    })


@dashboard_bp.route('/api/metrics-history')
@require_auth
def metrics_history():
    try:
        if os.path.exists(METRICS_HISTORY_FILE):
            with open(METRICS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
    except Exception:
        pass
    return jsonify([])


