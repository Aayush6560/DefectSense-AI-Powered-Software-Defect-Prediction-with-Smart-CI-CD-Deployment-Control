import os
import re
import time
from collections import deque
from pathlib import Path

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None

ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# EXPANDED KNOWLEDGE BASE  (was 15 entries -> now 49)
# ──────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        'id': 'kb_1',
        'text': 'Cyclomatic complexity measures the number of independent execution paths in a function. Values above 10 usually justify refactoring and additional tests. Values above 20 are critical.',
        'tags': {'cyclomatic', 'complexity', 'v(g)', 'control', 'paths'}
    },
    {
        'id': 'kb_2',
        'text': 'Halstead Volume estimates how much information is present in a program. Higher volume usually means denser logic and greater maintenance risk. Volumes above 1000 are a strong defect signal.',
        'tags': {'halstead', 'volume', 'information', 'density'}
    },
    {
        'id': 'kb_3',
        'text': 'Halstead Difficulty reflects the cognitive effort required to understand or write code. Large difficulty often signals brittle or complex logic. A difficulty above 15 usually needs attention.',
        'tags': {'halstead', 'difficulty', 'cognitive', 'effort'}
    },
    {
        'id': 'kb_4',
        'text': 'Lines of code alone is not a reliable defect signal, but unusually large modules (above 300 LOC) often deserve decomposition into smaller units with single responsibilities.',
        'tags': {'loc', 'lines', 'size', 'decomposition'}
    },
    {
        'id': 'kb_5',
        'text': 'Comment density can help maintainability, but comments should explain intent rather than restate code. Good naming and small functions matter more than comment volume alone.',
        'tags': {'comments', 'maintainability', 'intent'}
    },
    {
        'id': 'kb_6',
        'text': 'Static analysis and unit tests complement each other. Static metrics reveal structural risk while tests verify expected behavior and edge cases. Both are needed for reliable software.',
        'tags': {'static', 'analysis', 'tests', 'behavior'}
    },
    {
        'id': 'kb_7',
        'text': 'Security scanning should flag dangerous patterns such as eval, exec, shell=True, and unsafe deserialization before deployment. These are common root causes of runtime vulnerabilities.',
        'tags': {'security', 'eval', 'exec', 'shell', 'pickle'}
    },
    {
        'id': 'kb_8',
        'text': 'A good refactor path is to split high-complexity functions into smaller helpers, isolate I/O from logic, and test the extracted units separately. This reduces both complexity and defect probability.',
        'tags': {'refactor', 'helpers', 'io', 'logic', 'testing'}
    },
    {
        'id': 'kb_9',
        'text': 'Model confidence should be paired with recall and precision. High accuracy can still hide poor defect detection on the minority class. Use F1 score and AUC-ROC as primary evaluation metrics.',
        'tags': {'accuracy', 'recall', 'precision', 'imbalance', 'f1', 'roc'}
    },
    {
        'id': 'kb_10',
        'text': 'When working with defect prediction datasets, inspect missing markers and convert all numeric fields before training to avoid corrupt model inputs. SMOTE can handle class imbalance effectively.',
        'tags': {'dataset', 'missing', 'numeric', 'training', 'smote'}
    },
    {
        'id': 'kb_11',
        'text': 'Kubernetes deployments should define readiness and liveness probes, explicit images, and a stable service selector to avoid opaque rollout failures. Always set resource limits on containers.',
        'tags': {'kubernetes', 'deployment', 'readiness', 'liveness', 'service'}
    },
    {
        'id': 'kb_12',
        'text': 'Docker builds should keep the build context small with a .dockerignore file and should not include virtual environments or caches. Use multi-stage builds to minimize final image size.',
        'tags': {'docker', 'build', 'context', 'ignore', 'multistage'}
    },
    {
        'id': 'kb_13',
        'text': 'Context-grounded retrieval works best when the retriever stays close to real source material and the generator cites retrieved passages clearly. Hybrid search improves recall by combining vector and keyword signals.',
        'tags': {'rag', 'retriever', 'generator', 'source', 'hybrid'}
    },
    {
        'id': 'kb_14',
        'text': 'A repository-backed search can return explanations from code, workflows, and manifests without claiming external knowledge that is not present in the project.',
        'tags': {'repository', 'search', 'code', 'workflow', 'manifests'}
    },
    {
        'id': 'kb_15',
        'text': 'For defect-prone files, the first fix is usually to reduce branching complexity, simplify data flow, and add tests around the most error-prone paths.',
        'tags': {'defect', 'branching', 'tests', 'paths'}
    },
    # ── NEW ENTRIES ───────────────────────────────────────────
    {
        'id': 'kb_16',
        'text': 'Stacking ensemble models combine multiple base learners (RandomForest, XGBoost, GradientBoosting) with a meta-learner (LogisticRegression) to reduce variance and improve defect detection.',
        'tags': {'stacking', 'ensemble', 'randomforest', 'xgboost', 'metalearner'}
    },
    {
        'id': 'kb_17',
        'text': 'SHAP (SHapley Additive exPlanations) values explain individual predictions by attributing each feature a positive or negative contribution. Positive SHAP = increases defect risk; negative SHAP = reduces it.',
        'tags': {'shap', 'explainability', 'feature', 'contribution', 'xai'}
    },
    {
        'id': 'kb_18',
        'text': 'Class imbalance in defect datasets means defect-prone files are the minority class (often 10-20%). Techniques like SMOTE, class_weight=balanced, and threshold tuning improve recall on the minority class.',
        'tags': {'imbalance', 'smote', 'minority', 'threshold', 'recall'}
    },
    {
        'id': 'kb_19',
        'text': 'Halstead Estimated Bugs (b) directly approximates the number of latent defects in a module using the formula b = V / 3000. Values above 0.5 are a strong defect signal worth investigating.',
        'tags': {'halstead', 'bugs', 'estimated', 'latent', 'b'}
    },
    {
        'id': 'kb_20',
        'text': 'Branch count and decision points are strongly correlated with defect density. Every additional if/elif/for/while branch is a new test case that must be covered to prevent regression.',
        'tags': {'branch', 'decision', 'coverage', 'test', 'regression'}
    },
    {
        'id': 'kb_21',
        'text': 'Flask routes should be kept thin: move business logic into service modules, not directly inside route functions. This keeps cyclomatic complexity low and improves testability.',
        'tags': {'flask', 'routes', 'service', 'complexity', 'testability'}
    },
    {
        'id': 'kb_22',
        'text': 'Feature extraction for defect prediction should include Halstead metrics, cyclomatic complexity, LOC, comment ratio, and branch count. Radon is a reliable Python library for computing these metrics.',
        'tags': {'feature', 'extraction', 'radon', 'halstead', 'cyclomatic'}
    },
    {
        'id': 'kb_23',
        'text': 'Decision threshold tuning is critical for imbalanced datasets. The optimal threshold is found by maximizing F1 score on the validation set using the precision-recall curve, not by defaulting to 0.5.',
        'tags': {'threshold', 'f1', 'precision', 'recall', 'validation', 'tuning'}
    },
    {
        'id': 'kb_24',
        'text': 'High nesting depth (more than 3 levels of indentation) is a strong indicator of complex control flow. Flatten nested conditionals using early returns and guard clauses to reduce cognitive load.',
        'tags': {'nesting', 'indentation', 'guard', 'early', 'return', 'flatten'}
    },
    {
        'id': 'kb_25',
        'text': 'God functions that handle too many responsibilities at once are a primary defect source. A function should do exactly one thing and be short enough to fully understand in under 30 seconds.',
        'tags': {'god', 'function', 'responsibility', 'single', 'short'}
    },
    {
        'id': 'kb_26',
        'text': 'Kubernetes service selectors must exactly match pod labels. A mismatch causes the service to have zero endpoints, making the app unreachable even if pods are running and healthy.',
        'tags': {'kubernetes', 'selector', 'labels', 'endpoints', 'service', 'mismatch'}
    },
    {
        'id': 'kb_27',
        'text': 'Python virtual environments (.venv, env/) should never be included in Docker build context. Use a .dockerignore file to exclude them and keep image size small.',
        'tags': {'docker', 'venv', 'dockerignore', 'image', 'size'}
    },
    {
        'id': 'kb_28',
        'text': 'Ansible playbooks automate infrastructure provisioning. For DefectSense, Ansible can automate Docker image builds, Kubernetes manifest deployment, and health checks in a single idempotent playbook.',
        'tags': {'ansible', 'playbook', 'automation', 'provisioning', 'idempotent'}
    },
    {
        'id': 'kb_29',
        'text': 'Metric loc (lines of code) is the total number of lines in a file. Higher loc usually means larger review surface and potentially higher maintenance risk.',
        'tags': {'loc', 'lines of code', 'line count', 'size'}
    },
    {
        'id': 'kb_30',
        'text': 'Metric v(g) is cyclomatic complexity. It estimates the number of independent decision paths; higher values require more test cases.',
        'tags': {'v(g)', 'cyclomatic', 'complexity', 'decision paths'}
    },
    {
        'id': 'kb_31',
        'text': 'Metric ev(g) is essential complexity. It reflects how much structured control flow has degraded into hard-to-maintain logic.',
        'tags': {'ev(g)', 'essential complexity', 'structuredness'}
    },
    {
        'id': 'kb_32',
        'text': 'Metric iv(g) is design complexity (average complexity across functions in this implementation). It indicates average per-function branching burden.',
        'tags': {'iv(g)', 'design complexity', 'average complexity'}
    },
    {
        'id': 'kb_33',
        'text': 'Metric n is Halstead program length (N), derived from total operators and operands. Larger values mean more tokens to read and reason about.',
        'tags': {'n', 'halstead length', 'program length', 'tokens'}
    },
    {
        'id': 'kb_34',
        'text': 'Metric v is Halstead volume. It approximates the information content of code and tends to grow with logic density.',
        'tags': {'v', 'halstead volume', 'information content'}
    },
    {
        'id': 'kb_35',
        'text': 'Metric l is Halstead level. Higher level means code is easier to express; low level often points to difficult implementation details.',
        'tags': {'l', 'halstead level', 'implementation level'}
    },
    {
        'id': 'kb_36',
        'text': 'Metric d is Halstead difficulty. It estimates how hard code is to understand or implement. High values signal cognitive load.',
        'tags': {'d', 'halstead difficulty', 'cognitive load'}
    },
    {
        'id': 'kb_37',
        'text': 'Metric i is Halstead intelligence/content. It combines level and volume and approximates useful information present in the implementation.',
        'tags': {'i', 'halstead intelligence', 'content metric'}
    },
    {
        'id': 'kb_38',
        'text': 'Metric e is Halstead effort. It estimates implementation effort and often rises with difficulty and volume.',
        'tags': {'e', 'halstead effort', 'implementation effort'}
    },
    {
        'id': 'kb_39',
        'text': 'Metric b is Halstead estimated bugs (V/3000). It is a rough latent defect estimate used as a risk signal.',
        'tags': {'b', 'estimated bugs', 'halstead bugs', 'defect estimate'}
    },
    {
        'id': 'kb_40',
        'text': 'Metric t is Halstead time (E/18). It gives a rough estimate of time needed to implement or understand the code.',
        'tags': {'t', 'halstead time', 'effort time'}
    },
    {
        'id': 'kb_41',
        'text': 'Metric lOCode counts logical code lines (non-blank, non-comment lines). It is a cleaner size metric than raw loc.',
        'tags': {'lOCode', 'logical code lines', 'code lines'}
    },
    {
        'id': 'kb_42',
        'text': 'Metric lOComment counts comment-only lines. It helps track documentation presence, but quality of comments matters more than count.',
        'tags': {'lOComment', 'comment lines', 'documentation'}
    },
    {
        'id': 'kb_43',
        'text': 'Metric lOBlank counts blank lines. It helps describe layout/readability but is not a strong defect predictor alone.',
        'tags': {'lOBlank', 'blank lines', 'spacing'}
    },
    {
        'id': 'kb_44',
        'text': 'Metric locCodeAndComment counts mixed lines containing both code and inline comments.',
        'tags': {'locCodeAndComment', 'inline comments', 'mixed lines'}
    },
    {
        'id': 'kb_45',
        'text': 'Metric uniq_Op counts unique operators. High values can indicate diverse control and expression styles.',
        'tags': {'uniq_Op', 'unique operators', 'operator variety'}
    },
    {
        'id': 'kb_46',
        'text': 'Metric uniq_Opnd counts unique operands (identifiers/literals). Larger values can indicate broader state/data usage.',
        'tags': {'uniq_Opnd', 'unique operands', 'identifier variety'}
    },
    {
        'id': 'kb_47',
        'text': 'Metric total_Op counts total operators used in the file. It contributes to Halstead length and volume.',
        'tags': {'total_Op', 'total operators', 'operator count'}
    },
    {
        'id': 'kb_48',
        'text': 'Metric total_Opnd counts total operands used in the file. It contributes to Halstead length and volume.',
        'tags': {'total_Opnd', 'total operands', 'operand count'}
    },
    {
        'id': 'kb_49',
        'text': 'Metric branchCount counts branch points from if/for/while/except and boolean decisions. More branches usually require more tests.',
        'tags': {'branchCount', 'branch count', 'decision points', 'test paths'}
    },
]

# ──────────────────────────────────────────────────────────────
# GLOBALS
# ──────────────────────────────────────────────────────────────
_response_counter = 0
_chroma_client = None
_collection = None
_conversation_memory = {}
_repeat_tracker = {}

NOISE_TERMS = {
    'defectsense', 'context', 'chat', 'local', 'software', 'engineering', 'kb',
    'file', 'analyzed', 'successfully', 'answer', 'questions', 'risk', 'factors',
    'model', 'works', 'metric', 'meanings', 'you', 'ai', 'ctx'
}

# Minimum similarity score to include a KB result (filters irrelevant matches)
MIN_SIMILARITY_SCORE = 0.30
METRIC_CODES = {
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
    'locode', 'locomment', 'loblank', 'loccodeandcomment',
    'uniq_op', 'uniq_opnd', 'total_op', 'total_opnd', 'branchcount'
}


# ──────────────────────────────────────────────────────────────
# TOKENIZATION
# ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', text.lower())


def _query_tokens(question: str) -> list[str]:
    return _tokenize(question)


# ──────────────────────────────────────────────────────────────
# CHROMADB COLLECTION
# ──────────────────────────────────────────────────────────────

def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    if chromadb is None or embedding_functions is None:
        return None

    db_path = ROOT / 'data' / 'chroma_db'
    db_path.mkdir(parents=True, exist_ok=True)

    try:
        _chroma_client = chromadb.PersistentClient(path=str(db_path))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        _collection = _chroma_client.get_or_create_collection(
            name='defect_knowledge',
            embedding_function=embedding_fn,
        )
    except Exception:
        # If semantic dependencies are unavailable at runtime,
        # keep chat working with lexical fallback.
        _collection = None
        return None

    # Sync KB: add missing entries, update existing ones
    existing_ids = set(_collection.get()['ids']) if _collection.count() > 0 else set()
    new_items = [item for item in KNOWLEDGE_BASE if item['id'] not in existing_ids]
    if new_items:
        _collection.add(
            documents=[item['text'] for item in new_items],
            ids=[item['id'] for item in new_items],
            metadatas=[{'tags': ','.join(sorted(item['tags']))} for item in new_items],
        )

    return _collection


# ──────────────────────────────────────────────────────────────
# HYBRID SEARCH  (vector + keyword re-ranking)
# NEW: was pure vector only — now combines both signals
# ──────────────────────────────────────────────────────────────

def _keyword_score(query: str, text: str) -> float:
    """BM25-lite: count query token overlaps in KB text."""
    query_tokens = set(_tokenize(query))
    text_tokens = _tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    hits = sum(1 for t in text_tokens if t in query_tokens)
    return hits / (len(text_tokens) ** 0.5)  # length-normalised


def _exact_metric_key(query: str) -> str:
    q = query.lower()
    if 'v(g)' in q:
        return 'v(g)'
    if 'ev(g)' in q:
        return 'ev(g)'
    if 'iv(g)' in q:
        return 'iv(g)'
    tokens = set(_tokenize(query))
    for key in METRIC_CODES:
        if key in tokens:
            return key
    return ''


def _metric_boost_entries(query: str) -> list[dict]:
    metric_key = _exact_metric_key(query)
    if not metric_key:
        return []

    boosted = []
    for item in KNOWLEDGE_BASE:
        text_l = item.get('text', '').lower()
        tags_l = {t.lower() for t in item.get('tags', set())}
        if metric_key in tags_l or f'metric {metric_key}' in text_l:
            boosted.append({
                'id': item['id'],
                'text': item.get('text', ''),
                'score': 1.0,
                'vec_score': 1.0,
                'kw_score': 1.0,
            })
    return boosted


def _search_knowledge_entries(query: str, n_results: int = 4) -> list[dict]:
    global _collection
    collection = _get_collection()
    boosted = _metric_boost_entries(query)

    # Fast-profile fallback: lexical retrieval when semantic stack is unavailable.
    if collection is None:
        query_l = query.lower()
        entries = []
        for item in KNOWLEDGE_BASE:
            text = item.get('text', '')
            vec_score = _keyword_score(query, text)
            tag_hits = sum(1 for tag in item.get('tags', set()) if tag in query_l)
            tag_score = min(tag_hits * 0.1, 0.3)
            hybrid = round(min(vec_score + tag_score, 1.0), 4)
            if hybrid < MIN_SIMILARITY_SCORE:
                continue
            entries.append({
                'id': item['id'],
                'text': text,
                'score': hybrid,
                'vec_score': vec_score,
                'kw_score': vec_score,
            })

        entries.sort(key=lambda x: x['score'], reverse=True)
        if boosted:
            merged = {item['id']: item for item in boosted}
            for item in entries:
                merged.setdefault(item['id'], item)
            return list(merged.values())[:n_results]
        return entries[:n_results]

    # Pull more candidates than needed, then re-rank with keyword signal
    fetch_n = min(n_results * 2, len(KNOWLEDGE_BASE))
    try:
        results = collection.query(query_texts=[query], n_results=fetch_n)
    except Exception:
        # Runtime query failure (e.g., embedding backend issue) -> lexical fallback
        _collection = None
        return _search_knowledge_entries(query, n_results=n_results)

    ids        = results.get('ids',       [[]])[0]
    documents  = results.get('documents', [[]])[0]
    distances  = results.get('distances', [[]])[0]

    entries = []
    for i, doc in enumerate(documents):
        vec_score = round(float(1.0 - distances[i]), 4) if i < len(distances) and distances[i] is not None else 0.0

        # Filter out low-relevance results immediately
        if vec_score < MIN_SIMILARITY_SCORE:
            continue

        kw_score  = _keyword_score(query, doc)
        # Hybrid: 70% vector, 30% keyword
        hybrid    = round(0.70 * vec_score + 0.30 * kw_score, 4)

        entries.append({
            'id':        ids[i] if i < len(ids) else f'kb_{i}',
            'text':      doc,
            'score':     hybrid,
            'vec_score': vec_score,
            'kw_score':  kw_score,
        })

    # Sort by hybrid score descending, take top n_results
    entries.sort(key=lambda x: x['score'], reverse=True)
    if boosted:
        merged = {item['id']: item for item in boosted}
        for item in entries:
            merged.setdefault(item['id'], item)
        return list(merged.values())[:n_results]
    return entries[:n_results]


# ──────────────────────────────────────────────────────────────
# CONTEXT-AWARE QUERY REFORMULATION
# NEW: uses conversation history to improve retrieval
# ──────────────────────────────────────────────────────────────

def _reformulate_query(question: str, filename: str, top_features: list) -> str:
    """
    Combines current question with recent conversation context and top features
    to produce a richer retrieval query.
    """
    parts = [question.strip()]

    # Add top feature names as retrieval signals
    if top_features:
        feature_names = ' '.join(f[0] for f in top_features[:3])
        parts.append(feature_names)

    # Add last question from memory if it overlaps significantly
    history = _get_recent_question(filename)
    if history and _overlap_ratio(question, history) < 0.6:
        parts.append(history)

    if filename:
        parts.append(filename)

    return ' '.join(parts).strip()


def search_knowledge_base(query: str, n_results: int = 4) -> str:
    entries = _search_knowledge_entries(query, n_results=n_results)
    if not entries:
        return 'No relevant knowledge-base matches found above confidence threshold.'
    return '\n\n'.join(f"[{item['id']}]\n{item['text']}" for item in entries)


def get_context(query: str, n_results: int = 4) -> str:
    return search_knowledge_base(query, n_results=n_results)


# ──────────────────────────────────────────────────────────────
# STREAMING EXPLANATION ENTRY POINT
# ──────────────────────────────────────────────────────────────

def generate_ai_explanation(prediction_result: dict, question: str):
    """
    Generate a streaming answer grounded in ChromaDB knowledge and prediction data.
    Uses conversation-aware query reformulation for better retrieval.
    """
    prob         = prediction_result.get('probability', 0)
    label        = prediction_result.get('label', 'Unknown')
    decision_threshold = float(prediction_result.get('decision_threshold', 0.5) or 0.5)
    top_features = prediction_result.get('top_features', [])
    metrics      = prediction_result.get('metrics', {})
    summary      = prediction_result.get('summary', {})
    filename     = prediction_result.get('filename', 'this file')
    source_code  = prediction_result.get('source_code', '')

    # Context-aware query reformulation (NEW)
    rich_query = _reformulate_query(question, filename, top_features)

    use_full_context = _should_use_repo_context(question)
    entries   = _search_knowledge_entries(rich_query, n_results=4 if use_full_context else 3)
    context   = '\n\n'.join(f"[{e['id']}]\n{e['text']}" for e in entries)
    citations = [f"kb:{e['id']}" for e in entries]
    history_hint = _get_recent_question(filename)

    response = _build_response(
        question, prob, label, top_features,
        metrics, summary, filename, source_code,
        context, citations, history_hint, decision_threshold,
    )
    _remember_turn(filename, question, response)

    # Adaptive streaming: faster for short responses, normal for long
    words = response.split(' ')
    delay = 0.008 if len(words) < 60 else 0.012
    for i, word in enumerate(words):
        yield word + (' ' if i < len(words) - 1 else '')
        time.sleep(delay)


# ──────────────────────────────────────────────────────────────
# RESPONSE BUILDER  (cleaner routing, less fragile)
# ──────────────────────────────────────────────────────────────

def _build_response(question, prob, label, top_features, metrics, summary,
                    filename, source_code, context, citations=None, history_hint='', decision_threshold=0.5):
    global _response_counter
    _response_counter += 1

    question_clean = _sanitize_question(question)
    q = re.sub(r'[^a-z0-9()\s]+', ' ', question_clean.lower())
    q = re.sub(r'\s+', ' ', q).strip()

    pct           = prob * 100
    threshold_pct = float(decision_threshold) * 100
    risk_band     = _risk_band(prob)
    decision_text = _classification_summary(label, pct, threshold_pct, risk_band)
    vg            = float(metrics.get('v(g)', 0) or 0)
    v             = float(metrics.get('v', 0) or 0)
    b             = float(metrics.get('b', 0) or 0)
    d             = float(metrics.get('d', 0) or 0)
    loc           = float(metrics.get('loc', 0) or 0)
    branch_count  = float(metrics.get('branchCount', 0) or 0)
    top_feat_name = top_features[0][0] if top_features else 'cyclomatic complexity'
    top_feat_val  = top_features[0][1] if top_features else 0
    source_excerpt = _extract_relevant_source_excerpt(question, source_code)
    context_excerpt = '\n'.join(context.splitlines()[:10]) if context else ''
    citation_block  = _build_citation_block(citations or [])
    seed = _variant_seed(question_clean, filename, label, str(_response_counter))

    # Repeat / noise guard
    repeat_count = _track_repeat(filename, question_clean)
    metric_intent = _is_metric_intent(q)
    repeat_note = ''
    if history_hint and _overlap_ratio(question_clean, history_hint) >= 0.75:
        repeat_note = 'Your question is similar to your last one, so I am approaching it differently. '
    if (not metric_intent and _is_low_signal_prompt(question_clean)) or repeat_count >= 3:
        return _build_prompt_guidance(filename, decision_text, repeat_count)

    # ── Intent routing ─────────────────────────────────────────

    if _is_smalltalk(q):
        return (
            f"Hi. {filename} {decision_text}. "
            'Ask me: why is it risky, what should I fix first, explain top metrics, '
            'show refactor suggestions, or which function is most risky.'
        )

    if any(p in q for p in ['what is defect', 'whats defect', 'meaning of defect', 'define defect']):
        return (
            f"A defect means code likely to contain bugs or fail in edge cases. "
            f"For {filename}, {decision_text}. "
            f"Main signal: {top_feat_name} ({abs(top_feat_val):.3f}). "
            f"Cyclomatic complexity is {vg:.0f}, LOC is {loc:.0f}."
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    if _is_deployment_question(q):
        return _build_deployment_response(filename, context_excerpt, citation_block, seed)

    if _is_function_risk_question(q):
        return _build_function_risk_response(filename, source_code, citation_block)

    if metric_intent:
        metric_name = _extract_metric_name(q)
        return _build_metric_explanation(metrics, filename, decision_text, metric_name, citation_block)

    if any(p in q for p in ['where is the error', 'where error', 'error location', 'which line', 'where is bug']):
        function_hint = ', '.join((summary.get('functions') or [])[:2]) or 'the most complex function'
        return (
            f"I cannot pinpoint exact runtime bug lines from static metrics, but the highest-risk area is around {function_hint}. "
            f"Top signal: {top_feat_name} ({abs(top_feat_val):.3f}). "
            f"Start code review here:\n{source_excerpt}"
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    if any(p in q for p in ['can u fix', 'can you fix', 'fix it']):
        plan = _build_action_plan(metrics, summary, top_features)
        hotspots = _estimate_function_hotspots(source_code)
        target = hotspots[0][0] if hotspots else 'the highest-risk function'
        return (
            f"Yes. Start with {target} and apply this order:\n"
            f"1. {plan[0]}\n2. {plan[1]}\n3. {plan[2]}\n4. {plan[3]}\n"
            "Paste that function and I can return a concrete refactored version."
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    if any(w in q for w in ['fix', 'improve', 'refactor', 'what should', 'recommend',
                             'what to do', 'next step', 'how to fix', 'what now']):
        plan = _build_action_plan(metrics, summary, top_features)
        return (
            f"{repeat_note}Action plan for {filename} ({risk_band}, {pct:.1f}% | model label: {label}, threshold {threshold_pct:.1f}%):\n"
            f"1. {plan[0]}\n2. {plan[1]}\n3. {plan[2]}\n4. {plan[3]}\n"
            f"Key metrics: complexity {vg:.0f}, volume {v:.0f}, LOC {loc:.0f}."
            + (f"\n\nKnowledge base:\n{context_excerpt}" if context_excerpt else '')
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    if any(w in q for w in ['why', 'reason', 'risk', 'risky', 'defect', 'bug', 'predict']):
        threshold_note = ''
        if str(label).lower() == 'clean' and prob >= 0.4:
            threshold_note = (
                f" Note: probability is elevated, but it remains below the model threshold "
                f"({pct:.1f}% < {threshold_pct:.1f}%), so final label stays Clean."
            )
        return (
            f"{repeat_note}{filename} {decision_text}.{threshold_note} "
            f"Strongest feature: {top_feat_name} ({abs(top_feat_val):.3f}). "
            f"Cyclomatic complexity: {vg:.0f}, Halstead Volume: {v:.0f}, "
            f"Halstead Difficulty: {d:.1f}, Estimated Bugs: {b:.3f}, LOC: {loc:.0f}."
            + (f"\nFile excerpt:\n{source_excerpt}" if source_excerpt else '')
            + (f"\n\nKnowledge base:\n{context_excerpt}" if context_excerpt else '')
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    if any(w in q for w in ['model', 'algorithm', 'train', 'ml', 'machine learning', 'stacking', 'shap']):
        return (
            f"DefectSense uses a stacking ensemble (RandomForest + XGBoost + GradientBoosting meta-learner) "
            f"trained on NASA PROMISE metrics. SHAP values explain each prediction. "
            f"Current result for {filename}: {decision_text}."
            + (f"\n\nKnowledge base:\n{context_excerpt}" if context_excerpt else '')
            + (f"\n\n{citation_block}" if citation_block else '')
        )

    # Default fallback
    return (
        f"For {filename}, current prediction: {decision_text}. "
        f"Strongest signal: {top_feat_name}. "
        + (f"File excerpt:\n{source_excerpt}" if source_excerpt else '')
        + (f"\n\nKnowledge base:\n{context_excerpt}" if context_excerpt else '')
        + (f"\n\n{citation_block}" if citation_block else '')
    )


# ──────────────────────────────────────────────────────────────
# RESPONSE BUILDERS  (decomposed from monolithic _build_response)
# ──────────────────────────────────────────────────────────────

def _build_deployment_response(filename: str, context_excerpt: str, citation_block: str, seed: int) -> str:
    plan = _build_deployment_plan()
    return (
        f"Deployment checklist for {filename}:\n"
        f"1. {plan[0]}\n2. {plan[1]}\n3. {plan[2]}\n4. {plan[3]}\n"
        "Order: cluster reachability → manifests → rollout status → service endpoint check."
        + (f"\n\nKnowledge base:\n{context_excerpt}" if context_excerpt else '')
        + (f"\n\n{citation_block}" if citation_block else '')
    )


def _build_function_risk_response(filename: str, source_code: str, citation_block: str) -> str:
    hotspots = _estimate_function_hotspots(source_code)
    if hotspots:
        lines = [
            f"{i + 1}. {name} (risk score {score:.1f}, {loc} lines)"
            for i, (name, score, loc) in enumerate(hotspots[:3])
        ]
        return (
            f"Most likely high-risk functions in {filename}:\n"
            + "\n".join(lines)
            + "\nRanked by branch density, nesting depth, and function size."
            + (f"\n\n{citation_block}" if citation_block else '')
        )
    return (
        f"I could not detect function blocks in {filename}. "
        "Upload the full source file and I will rank risky functions directly."
    )


# ──────────────────────────────────────────────────────────────
# CONVERSATION MEMORY
# ──────────────────────────────────────────────────────────────

def _remember_turn(filename: str, question: str, answer: str) -> None:
    key  = filename or 'global'
    convo = _conversation_memory.setdefault(key, deque(maxlen=6))
    convo.append({'q': question.strip(), 'a': answer[:300]})


def _get_recent_question(filename: str) -> str:
    key   = filename or 'global'
    convo = _conversation_memory.get(key)
    if not convo:
        return ''
    return convo[-1].get('q', '')


def _overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


# ──────────────────────────────────────────────────────────────
# INTENT DETECTION
# ──────────────────────────────────────────────────────────────

def _is_smalltalk(q: str) -> bool:
    compact = q.strip()
    if compact in {'hi', 'hello', 'hey', 'yo', 'sup', 'hii'}:
        return True
    return any(p in compact for p in ['hello ai', 'hi ai', 'hey ai'])


def _is_deployment_question(q: str) -> bool:
    terms = {
        'deploy', 'deployment', 'kubernetes', 'k8s', 'pipeline', 'ci', 'cd',
        'rollout', 'manifest', 'kubectl', 'docker build', 'container',
        'service endpoint', 'ansible', 'helm'
    }
    return any(t in q for t in terms)


def _is_function_risk_question(q: str) -> bool:
    tokens = set(_query_tokens(q))
    func_tokens = {'function', 'functions', 'funtion', 'fuction', 'func', 'method', 'methods'}
    risk_tokens = {'risk', 'risky', 'defect', 'defects', 'bug', 'bugs', 'error', 'errors'}
    direct = [
        'which function has high risk', 'which function has more risk',
        'high risk function', 'risky function', 'which function is risky',
        'most risky function', 'risky functions'
    ]
    return any(p in q for p in direct) or bool((tokens & func_tokens) and (tokens & risk_tokens))


def _is_metric_intent(q: str) -> bool:
    if q.strip() in {'explain', 'explain metric', 'explain metrics', 'metrics', 'metric'}:
        return True
    tokens = set(_query_tokens(q))
    if tokens & {'cyclomatic', 'halstead', 'loc', 'difficulty', 'volume', 'branchcount'}:
        return True
    if 'explain' in tokens and tokens & {'metric', 'metrics', 'complexity', 'bugs', 'lines'}:
        return True
    if _extract_metric_name(q):
        return True
    return bool(tokens & {'metric', 'metrics'})


def _should_use_repo_context(question: str) -> bool:
    keywords = {
        'repo', 'repository', 'project', 'pipeline', 'cicd', 'ci/cd', 'workflow',
        'docker', 'kubernetes', 'k8s', 'deployment', 'ansible', 'manifest',
        'model', 'train', 'extractor', 'rag', 'knowledge base', 'stacking', 'shap'
    }
    return any(k in question.lower() for k in keywords)


def _is_low_signal_prompt(question: str) -> bool:
    q_lower = question.lower().strip()
    valid_short = {
        'risk', 'risky', 'why risk', 'fix', 'top fixes', 'top 3 code fixes',
        'what to do', 'what now', 'how to fix', 'next step', 'next steps',
        'where is the error', 'where error', 'metrics', 'metric', 'explain metrics'
    }
    if q_lower in valid_short:
        return False
    tokens = _query_tokens(question)
    unique = len(set(tokens))
    if unique < 2:
        return True
    if len(question) > 120 and unique < 6:
        return True
    return False


def _track_repeat(filename: str, question: str) -> int:
    key    = filename or 'global'
    bucket = _repeat_tracker.setdefault(key, {})
    q_key  = question.lower().strip()
    count  = bucket.get(q_key, 0) + 1
    bucket[q_key] = count
    return count


# ──────────────────────────────────────────────────────────────
# METRIC EXTRACTION & EXPLANATION
# ──────────────────────────────────────────────────────────────

METRIC_DEFINITIONS = {
    'loc': {
        'title': 'Lines of code',
        'aliases': ['loc', 'lines of code', 'line count'],
        'meaning': 'Total lines in the file. Larger modules are harder to review and test.',
        'warn': 150,
        'critical': 300,
    },
    'v(g)': {
        'title': 'Cyclomatic complexity',
        'aliases': ['v(g)', 'vg', 'cyclomatic', 'cyclomatic complexity'],
        'meaning': 'Number of independent decision paths that tests must cover.',
        'warn': 10,
        'critical': 20,
    },
    'ev(g)': {
        'title': 'Essential complexity',
        'aliases': ['ev(g)', 'essential complexity'],
        'meaning': 'How much control flow has become structurally difficult to simplify.',
    },
    'iv(g)': {
        'title': 'Design complexity',
        'aliases': ['iv(g)', 'design complexity', 'average complexity'],
        'meaning': 'Average per-function complexity in the implementation.',
    },
    'n': {
        'title': 'Halstead program length',
        'aliases': ['n', 'halstead length', 'program length'],
        'meaning': 'Total operator and operand occurrences in Halstead terms.',
    },
    'v': {
        'title': 'Halstead volume',
        'aliases': ['v', 'halstead volume', 'volume'],
        'meaning': 'Information content of the code; high values indicate denser logic.',
        'warn': 500,
        'critical': 1000,
    },
    'l': {
        'title': 'Halstead level',
        'aliases': ['l', 'halstead level', 'level'],
        'meaning': 'Expression level of implementation; lower values usually mean harder code.',
    },
    'd': {
        'title': 'Halstead difficulty',
        'aliases': ['d', 'halstead difficulty', 'difficulty'],
        'meaning': 'Estimated cognitive effort required to understand or implement the code.',
        'warn': 10,
        'critical': 20,
    },
    'i': {
        'title': 'Halstead intelligence',
        'aliases': ['i', 'halstead intelligence', 'content'],
        'meaning': 'Derived Halstead signal for useful information content.',
    },
    'e': {
        'title': 'Halstead effort',
        'aliases': ['e', 'halstead effort', 'effort'],
        'meaning': 'Estimated effort to implement or reason about code.',
    },
    'b': {
        'title': 'Estimated bugs',
        'aliases': ['b', 'estimated bugs', 'bug estimate', 'halstead bugs'],
        'meaning': 'Approximate latent defects (Halstead V/3000).',
        'warn': 0.3,
        'critical': 0.7,
    },
    't': {
        'title': 'Halstead time',
        'aliases': ['t', 'halstead time', 'time'],
        'meaning': 'Approximate time from Halstead effort (E/18).',
    },
    'lOCode': {
        'title': 'Logical code lines',
        'aliases': ['locode', 'logical code lines', 'code lines'],
        'meaning': 'Non-blank, non-comment code lines.',
    },
    'lOComment': {
        'title': 'Comment lines',
        'aliases': ['locomment', 'comment lines', 'comment count'],
        'meaning': 'Lines that are comments only.',
    },
    'lOBlank': {
        'title': 'Blank lines',
        'aliases': ['loblank', 'blank lines'],
        'meaning': 'Blank/whitespace lines in the file.',
    },
    'locCodeAndComment': {
        'title': 'Code and inline comment lines',
        'aliases': ['loccodeandcomment', 'code and comment', 'inline comment lines'],
        'meaning': 'Lines containing both code and inline comments.',
    },
    'uniq_Op': {
        'title': 'Unique operators',
        'aliases': ['uniq_op', 'unique operators'],
        'meaning': 'Distinct operator types used in the file.',
    },
    'uniq_Opnd': {
        'title': 'Unique operands',
        'aliases': ['uniq_opnd', 'unique operands'],
        'meaning': 'Distinct operands (names/literals) used in the file.',
    },
    'total_Op': {
        'title': 'Total operators',
        'aliases': ['total_op', 'total operators'],
        'meaning': 'Total operator occurrences used in the file.',
    },
    'total_Opnd': {
        'title': 'Total operands',
        'aliases': ['total_opnd', 'total operands'],
        'meaning': 'Total operand occurrences used in the file.',
    },
    'branchCount': {
        'title': 'Branch count',
        'aliases': ['branchcount', 'branch count', 'branches'],
        'meaning': 'Branching decision points from conditionals/loops/handlers.',
        'warn': 15,
        'critical': 30,
    },
}


def _contains_alias(q: str, alias: str) -> bool:
    alias = alias.strip().lower()
    if not alias:
        return False
    if re.fullmatch(r'[a-z0-9_]+', alias):
        return re.search(rf'\b{re.escape(alias)}\b', q) is not None
    return alias in q

def _extract_metric_name(q: str) -> str:
    for metric_key, spec in METRIC_DEFINITIONS.items():
        aliases = [metric_key.lower(), spec['title'].lower()] + [a.lower() for a in spec.get('aliases', [])]
        if any(_contains_alias(q, alias) for alias in aliases):
            return metric_key
    return ''


def _build_metric_explanation(metrics: dict, filename: str, decision_text: str, metric_name: str, citation_block: str) -> str:
    def _severity_label(value: float, warn, critical) -> str:
        if warn is None or critical is None:
            return 'No fixed threshold'
        if value >= critical:
            return 'Critical'
        if value >= warn:
            return 'Warning'
        return 'OK'

    def _format_value(value: float) -> str:
        if abs(value) >= 100 or value.is_integer():
            return f'{value:.0f}'
        if abs(value) >= 1:
            return f'{value:.2f}'
        return f'{value:.3f}'

    if metric_name and metric_name in METRIC_DEFINITIONS:
        spec = METRIC_DEFINITIONS[metric_name]
        value = float(metrics.get(metric_name, 0) or 0)
        status = _severity_label(value, spec.get('warn'), spec.get('critical'))
        threshold_hint = ''
        if spec.get('warn') is not None and spec.get('critical') is not None:
            threshold_hint = f" Typical thresholds: warning >= {spec['warn']}, critical >= {spec['critical']}."
        text = (
            f"{filename} {decision_text}.\n"
            f"Metric: {spec['title']} ({metric_name})\n"
            f"Current value: {_format_value(value)} ({status})\n"
            f"Meaning: {spec['meaning']}{threshold_hint}"
        )
    else:
        lines = [f"Metric glossary for {filename} ({decision_text}):"]
        ordered_metrics = [
            'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
            'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
            'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount'
        ]
        for key in ordered_metrics:
            spec = METRIC_DEFINITIONS[key]
            value = float(metrics.get(key, 0) or 0)
            status = _severity_label(value, spec.get('warn'), spec.get('critical'))
            lines.append(f"- {spec['title']} ({key}): {_format_value(value)} | {status} | {spec['meaning']}")
        lines.append('Ask: explain metric <name>, for example explain metric v(g), explain metric uniq_Op, or explain metric locCodeAndComment.')
        text = '\n'.join(lines)

    return text + (f"\n\n{citation_block}" if citation_block else '')


# ──────────────────────────────────────────────────────────────
# ACTION PLAN BUILDER  (priority order, NOT randomized)
# ──────────────────────────────────────────────────────────────

def _build_action_plan(metrics: dict, summary: dict, top_features: list) -> list[str]:
    """
    Returns 4 prioritized actions based on actual metric values.
    Actions are in fixed priority order — NOT rotated randomly.
    """
    vg           = float(metrics.get('v(g)',        0) or 0)
    v            = float(metrics.get('v',           0) or 0)
    d            = float(metrics.get('d',           0) or 0)
    loc          = float(metrics.get('loc',         0) or 0)
    branch_count = float(metrics.get('branchCount', 0) or 0)
    functions    = summary.get('functions', []) or []
    top_feature  = top_features[0][0] if top_features else 'v(g)'

    actions = []

    if vg >= 10 or branch_count >= 8:
        target = ', '.join(functions[:2]) if functions else 'the most nested function'
        actions.append(
            f"Reduce cyclomatic complexity first: split {target} into smaller helpers "
            f"and flatten nested conditionals using early returns (current v(g)={vg:.0f})."
        )

    if v >= 500 or d >= 10:
        actions.append(
            f"Reduce cognitive load: extract calculation blocks into named functions "
            f"and replace magic constants with clear variables (Volume={v:.0f}, Difficulty={d:.1f})."
        )

    if loc >= 150:
        actions.append(
            f"Split module: move file, parsing, and I/O logic into separate modules so each "
            f"has one responsibility (current LOC={loc:.0f})."
        )

    actions.append(
        f"Write targeted tests for the top risk signal ({top_feature}): cover happy path, "
        "edge cases, and failure paths before and after refactoring."
    )

    if len(actions) < 4:
        actions.append(
            "Run a complexity scan after each change and verify the defect probability drops "
            "in DefectSense before merging."
        )

    while len(actions) < 4:
        actions.append("Run an incremental cleanup pass and verify risk score after each change.")

    return actions[:4]


def _build_deployment_plan() -> list[str]:
    return [
        "Verify cluster connection: run `kubectl cluster-info` and `kubectl get nodes`.",
        "Re-apply manifests: `kubectl apply -f k8s/` then check events with `kubectl get events --sort-by=.lastTimestamp`.",
        "Check rollout health: `kubectl rollout status deployment/proj3-defectsense --timeout=180s` and inspect failing pods.",
        "Validate service routing: `kubectl get svc,ep proj3-defectsense -o wide` and test the app URL.",
    ]


# ──────────────────────────────────────────────────────────────
# FUNCTION HOTSPOT DETECTION
# ──────────────────────────────────────────────────────────────

def _estimate_function_hotspots(source_code: str, max_items: int = 5) -> list[tuple[str, float, int]]:
    if not source_code.strip():
        return []

    lines = source_code.splitlines()
    blocks = []
    current_name = current_start = current_indent = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent   = len(line) - len(stripped)

        if stripped.startswith(('def ', 'async def ')):
            if current_name is not None:
                blocks.append((current_name, current_start, i))
            match = re.match(r'(?:async\s+def|def)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', stripped)
            current_name  = match.group(1) if match else f'function_at_line_{i + 1}'
            current_start = i
            current_indent = indent
        elif current_name is not None and stripped and current_indent is not None and indent <= current_indent:
            blocks.append((current_name, current_start, i))
            current_name = current_start = current_indent = None

    if current_name is not None:
        blocks.append((current_name, current_start, len(lines)))

    ranked = []
    for name, start, end in blocks:
        block_text = '\n'.join(lines[start:end])
        func_loc   = max(1, end - start)
        branches   = len(re.findall(r'\b(if|elif|for|while|try|except|and|or)\b', block_text))
        nesting    = len(re.findall(r'\n\s{8,}\S', block_text))
        score      = (branches * 2.5) + (func_loc * 0.12) + (nesting * 0.8)
        ranked.append((name, round(score, 1), func_loc))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:max_items]


# ──────────────────────────────────────────────────────────────
# SOURCE EXCERPT EXTRACTION
# ──────────────────────────────────────────────────────────────

def _extract_relevant_source_excerpt(question: str, source_code: str, max_lines: int = 18) -> str:
    if not source_code.strip():
        return 'Source code is unavailable.'

    lines = source_code.splitlines()
    question_tokens = [t for t in _tokenize(question) if len(t) > 2]
    scored_blocks = []

    current_start = current_indent = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent   = len(line) - len(stripped)
        if stripped.startswith(('def ', 'class ', 'async def ')):
            if current_start is not None:
                scored_blocks.append((current_start, i, '\n'.join(lines[current_start:i]).strip()))
            current_start  = i
            current_indent = indent
        elif current_start is not None and stripped and current_indent is not None and indent <= current_indent:
            scored_blocks.append((current_start, i, '\n'.join(lines[current_start:i]).strip()))
            current_start = current_indent = None

    if current_start is not None:
        scored_blocks.append((current_start, len(lines), '\n'.join(lines[current_start:]).strip()))

    ranked = []
    for start, end, block in scored_blocks:
        if not block:
            continue
        first = block.splitlines()[0].lower() if block.splitlines() else ''
        score = 7 if first.startswith(('def ', 'async def ')) else (2 if first.startswith('class ') else 0)
        score += sum(3 for t in question_tokens if t in block.lower())
        if score:
            ranked.append((score, end - start, block))

    if ranked:
        ranked.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        excerpt = ranked[0][2]
        excerpt_lines = excerpt.splitlines()
        return '\n'.join(excerpt_lines[:max_lines]) if len(excerpt_lines) > max_lines else excerpt

    # Fallback: scored window search
    windows = []
    for i, line in enumerate(lines):
        score = sum(3 for t in question_tokens if t in line.lower())
        score += 2 if line.lstrip().startswith(('def ', 'class ', 'async def ')) else 0
        if score:
            start = max(0, i - 2)
            end   = min(len(lines), i + 5)
            windows.append((score, '\n'.join(lines[start:end]).strip()))

    if windows:
        windows.sort(key=lambda x: x[0], reverse=True)
        excerpt_lines = windows[0][1].splitlines()
        return '\n'.join(excerpt_lines[:max_lines])

    return '\n'.join(lines[:max_lines]).strip() or 'Source code is unavailable.'


# ──────────────────────────────────────────────────────────────
# MISC HELPERS
# ──────────────────────────────────────────────────────────────

def _build_citation_block(citations: list[str]) -> str:
    return ('Sources: ' + ', '.join(citations)) if citations else ''


def _build_prompt_guidance(filename: str, decision_text: str, repeat_count: int) -> str:
    hint = 'You repeated the same prompt several times. ' if repeat_count >= 3 else ''
    return (
        f"{hint}{filename} is currently {decision_text}. "
        'Ask a focused question: '
        '1) why is risk high, '
        '2) top 3 code fixes, '
        '3) kubernetes deploy checklist, '
        '4) explain metric <name>, or '
        '5) which function is most risky.'
    )


def _risk_band(prob: float) -> str:
    if prob >= 0.7:
        return 'High-Risk'
    if prob >= 0.4:
        return 'Medium-Risk'
    return 'Low-Risk'


def _classification_summary(label: str, pct: float, threshold_pct: float, risk_band: str) -> str:
    return (
        f"is {risk_band} at {pct:.1f}% probability "
        f"(model label: {label}, threshold: {threshold_pct:.1f}%)"
    )


def _sanitize_question(question: str) -> str:
    lines = question.strip().splitlines()
    filtered = []
    for line in lines:
        lower = line.lower()
        if not any(p in lower for p in [
            'defectsense', 'local software engineering', 'context:', 'chat with me',
            'kb:', 'file analyzed', 'successfully', 'answer questions', 'model:',
            'risk factors', 'metrics:', '---', '===', '<<<', '>>>'
        ]) and line.strip():
            filtered.append(line)
    return '\n'.join(filtered).strip()


def _variant_seed(*parts: str) -> int:
    return sum(ord(ch) for ch in '|'.join(parts)) % 997


def _pick(options: list[str], seed: int) -> str:
    return options[seed % len(options)] if options else ''