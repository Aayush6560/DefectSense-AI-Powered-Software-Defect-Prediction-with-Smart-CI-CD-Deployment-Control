import json
import time
import hashlib
from flask import Blueprint, request, jsonify, render_template, Response, stream_with_context, make_response
from werkzeug.utils import secure_filename
from extractor import extract_metrics, get_code_summary, get_risk_breakdown
from ml.predict import predict_file, get_model_meta, is_model_loaded
from rag_chat import generate_ai_explanation, search_knowledge_base
from pipeline import run_pipeline_stream, get_pipeline_history
from auth import require_auth, increment_prediction_count

main = Blueprint('main', __name__)

_MAX_PREDICTIONS = 200
_predictions: dict = {}


def _evict_oldest(store: dict, max_size: int) -> None:
    if len(store) >= max_size:
        oldest = sorted(store.items(), key=lambda x: x[1].get('_ts', 0))
        for key, _ in oldest[:max(1, len(store) - max_size + 1)]:
            del store[key]


def _sse_response(generator_fn):
    return Response(
        stream_with_context(generator_fn()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@main.route('/')
def index():
    return render_template('login.html')


@main.route('/app')
def app_page():
    return render_template('index.html')


@main.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    filename = secure_filename(f.filename or '')

    if not filename or not filename.endswith('.py'):
        return jsonify({'error': 'Only .py files are supported'}), 400

    try:
        source_code = f.read().decode('utf-8', errors='ignore')
    except Exception:
        return jsonify({'error': 'Failed to read file'}), 400

    if len(source_code.strip()) < 10:
        return jsonify({'error': 'File is empty or too small to analyze'}), 400

    if len(source_code) > 5 * 1024 * 1024:
        return jsonify({'error': 'File exceeds 5MB limit'}), 413

    try:
        metrics = extract_metrics(source_code)
        summary = get_code_summary(source_code)
        prediction = predict_file(metrics, filename=filename)
        risks = get_risk_breakdown(metrics)
    except RuntimeError as e:
        return jsonify({'error': str(e), 'hint': 'Run: python train.py'}), 503
    except Exception as e:
        return jsonify({'error': 'Analysis failed. Check server logs for details.'}), 500

    username = request.current_user.get('sub', 'anonymous')
    _evict_oldest(_predictions, _MAX_PREDICTIONS)
    _predictions[username] = {
        '_ts': time.time(),
        'filename': filename,
        'source_code': source_code,
        'metrics': metrics,
        'summary': summary,
        'prediction': prediction,
        'risks': risks,
    }
    increment_prediction_count(username)

    return jsonify({
        'filename': filename,
        'summary': summary,
        'key_metrics': {
            'Cyclomatic Complexity': round(metrics.get('v(g)', 0), 2),
            'Lines of Code': round(metrics.get('loc', 0), 0),
            'Halstead Volume': round(metrics.get('v', 0), 2),
            'Branch Count': round(metrics.get('branchCount', 0), 0),
            'Unique Operators': round(metrics.get('uniq_Op', 0), 0),
            'Halstead Difficulty': round(metrics.get('d', 0), 2),
            'Halstead Bugs Est.': round(metrics.get('b', 0), 4),
            'Halstead Effort': round(metrics.get('e', 0), 2),
        },
        'probability': prediction['probability'],
        'raw_probability': prediction.get('raw_probability', prediction['probability']),
        'label': prediction['label'],
        'decision_threshold': prediction.get('decision_threshold', 0.5),
        'confidence_band': prediction.get('confidence_band', 'low'),
        'calibration': prediction.get('calibration', {}),
        'top_features': prediction['top_features'],
        'risks': risks,
        'model_meta': prediction.get('model_meta', {}),
    })


@main.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if len(question) > 2000:
        return jsonify({'error': 'Question too long (max 2000 characters)'}), 400

    username = request.current_user.get('sub', 'anonymous')
    prediction_data = _predictions.get(username)

    if not prediction_data:
        return jsonify({'error': 'Please upload and analyze a file first'}), 400

    full_context = {
        'filename': prediction_data['filename'],
        'source_code': prediction_data.get('source_code', '')[:3000],
        'probability': prediction_data['prediction']['probability'],
        'label': prediction_data['prediction']['label'],
        'decision_threshold': prediction_data['prediction'].get('decision_threshold', 0.5),
        'top_features': prediction_data['prediction']['top_features'],
        'metrics': prediction_data['metrics'],
        'summary': prediction_data['summary'],
        'model_meta': prediction_data['prediction'].get('model_meta', {}),
    }

    def stream():
        try:
            for chunk in generate_ai_explanation(full_context, question):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return _sse_response(stream)


@main.route('/api/pipeline', methods=['POST'])
@require_auth
def run_pipeline():
    username = request.current_user.get('sub', 'anonymous')
    prediction_data = _predictions.get(username)

    if not prediction_data:
        return jsonify({'error': 'Please upload and analyze a file first'}), 400

    def stream():
        try:
            for update in run_pipeline_stream(
                prediction_data['filename'],
                prediction_data.get('source_code', ''),
                prediction_data['metrics'],
                prediction_data['prediction'],
            ):
                yield f"data: {update}\n\n"
            yield 'data: {"type": "stream_end"}\n\n'
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return _sse_response(stream)


@main.route('/api/pipeline/history')
@require_auth
def pipeline_history():
    response = make_response(jsonify(get_pipeline_history()), 200)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


@main.route('/api/model-info')
@require_auth
def model_info():
    meta = get_model_meta()
    return jsonify({
        'loaded': is_model_loaded(),
        'meta': meta,
    })


@main.route('/api/rag-search', methods=['POST'])
@require_auth
def rag_search():
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'results': []}), 200
    context = search_knowledge_base(query, n_results=3)
    return jsonify({'context': context})