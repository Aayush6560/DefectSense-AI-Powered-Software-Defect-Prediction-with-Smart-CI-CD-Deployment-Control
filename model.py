import os
import pickle
import sys
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, 'ml', 'models')
LEGACY_MODELS_DIR = os.path.join(ROOT, 'models')

_model = None
_scaler = None
_feature_cols = None
_model_meta = None
_shap_explainer = None  # Cached real SHAP explainer


# ──────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ──────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _confidence_band(probability: float, threshold: float) -> str:
    """
    4-level confidence band instead of 3.
    Reduces overconfident labels near the decision boundary.
    """
    margin = abs(probability - threshold)
    if margin >= 0.35:
        return 'very_high'
    if margin >= 0.20:
        return 'high'
    if margin >= 0.10:
        return 'medium'
    return 'low'


def _is_orchestration_file(filename: str) -> bool:
    if not filename:
        return False
    name = filename.lower()
    orchestration_tokens = (
        'pipeline', 'deploy', 'docker', 'k8s', 'kube', 'ci', 'cd',
        'workflow', 'jenkins', 'compose', 'helm', 'makefile', 'ansible',
        'terraform', 'script', 'setup', 'install'
    )
    return any(token in name for token in orchestration_tokens)


# ──────────────────────────────────────────────────────────────
# CALIBRATION  (two-way, less aggressive)
# ──────────────────────────────────────────────────────────────

def _apply_probability_calibration(
    raw_probability: float,
    threshold: float,
    metrics: dict,
    filename: str
) -> tuple[float, dict]:
    """
    Two-way calibration:
      - Downward nudge for simple/orchestration files  (reduces false positives)
      - Upward nudge for genuinely complex/risky files (improves recall)

    All adjustments are conservative to avoid hurting overall accuracy.
    """
    adjustment = 0.0
    reasons = []

    # ── Downward adjustments ──────────────────────────────────

    # Orchestration files: structurally complex but not logically defect-prone
    if _is_orchestration_file(filename):
        adjustment -= 0.05          # was -0.08 in old version
        reasons.append('orchestration_filename')

    loc = float(metrics.get('loc', 0.0))
    branch_count = float(metrics.get('branchCount', 0.0))

    # Large LOC but low branching = procedural glue code
    if loc >= 250 and branch_count <= 20:
        adjustment -= 0.03          # was -0.04
        reasons.append('glue_code_profile')

    vg = float(metrics.get('v(g)', 0.0))
    diff = float(metrics.get('d', 0.0))
    bugs_est = float(metrics.get('b', 0.0))

    # Trivially simple module — small nudge only
    # KEY FIX: was -0.18 which destroyed recall on small files → now -0.06
    if vg <= 7 and branch_count <= 5 and loc <= 80 and diff <= 10 and bugs_est <= 0.2:
        adjustment -= 0.06
        reasons.append('low_complexity_profile')

    # ── Upward adjustments (NEW) ──────────────────────────────

    # Very high complexity is a strong defect signal — boost it
    if vg >= 20 and loc >= 200 and branch_count >= 30:
        adjustment += 0.04
        reasons.append('high_complexity_boost')

    # High Halstead bug estimate is a direct defect predictor
    if bugs_est >= 1.0:
        adjustment += 0.03
        reasons.append('high_bug_estimate_boost')

    calibrated = _clamp(raw_probability + adjustment, 0.001, 0.999)
    return calibrated, {
        'applied': bool(reasons),
        'reasons': reasons,
        'adjustment': float(adjustment),
        'raw_probability': float(raw_probability),
        'calibrated_probability': float(calibrated),
        'threshold': float(threshold),
    }


# ──────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE EXTRACTION
# ──────────────────────────────────────────────────────────────

def _get_feature_importances() -> np.ndarray:
    """
    Pulls feature importances in order of reliability:
    1. Direct (RandomForest, XGBoost, GBM)
    2. First tree-based estimator inside a stacking ensemble
    3. Zero array fallback (safe, no crash)
    """
    if hasattr(_model, 'feature_importances_'):
        return np.asarray(_model.feature_importances_, dtype=float)
    if hasattr(_model, 'estimators_') and len(_model.estimators_) > 0:
        for estimator in _model.estimators_:
            est = estimator[1] if isinstance(estimator, tuple) else estimator
            if hasattr(est, 'feature_importances_'):
                return np.asarray(est.feature_importances_, dtype=float)
    return np.zeros(len(_feature_cols), dtype=float)


# ──────────────────────────────────────────────────────────────
# NUMPY PICKLE COMPATIBILITY
# ──────────────────────────────────────────────────────────────

def _install_numpy_pickle_compat() -> None:
    """
    Aliases numpy._core → numpy.core so pickles saved on NumPy 2.x
    load cleanly on NumPy 1.x runtimes and vice versa.
    """
    try:
        import numpy.core as np_core
        import numpy.core.numeric as np_numeric
        sys.modules.setdefault('numpy._core', np_core)
        sys.modules.setdefault('numpy._core.numeric', np_numeric)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# MODEL LOADING  (lazy, single load, SHAP explainer cached)
# ──────────────────────────────────────────────────────────────

def _load_models() -> None:
    global _model, _scaler, _feature_cols, _model_meta, _shap_explainer
    if _model is not None:
        return

    load_dir = MODELS_DIR
    model_path = os.path.join(load_dir, 'stacking_model.pkl')
    if not os.path.exists(model_path):
        legacy_model_path = os.path.join(LEGACY_MODELS_DIR, 'stacking_model.pkl')
        if os.path.exists(legacy_model_path):
            load_dir = LEGACY_MODELS_DIR
            model_path = legacy_model_path

    if not os.path.exists(model_path):
        raise RuntimeError(
            "Model not trained yet. Run: python train.py\n"
            f"Expected: {model_path}"
        )

    _install_numpy_pickle_compat()

    with open(os.path.join(load_dir, 'stacking_model.pkl'), 'rb') as f:
        _model = pickle.load(f)
    with open(os.path.join(load_dir, 'scaler.pkl'), 'rb') as f:
        _scaler = pickle.load(f)
    with open(os.path.join(load_dir, 'feature_cols.pkl'), 'rb') as f:
        _feature_cols = pickle.load(f)
    with open(os.path.join(load_dir, 'model_meta.pkl'), 'rb') as f:
        _model_meta = pickle.load(f)

    # Build SHAP explainer once at load time (reused for every prediction)
    _shap_explainer = _build_shap_explainer()


def _build_shap_explainer():
    """
    Builds a real shap.TreeExplainer from the best available tree estimator.
    Returns None silently if shap is not installed — proxy fallback takes over.
    """
    try:
        import shap
        candidate = None
        if hasattr(_model, 'estimators_'):
            for estimator in _model.estimators_:
                est = estimator[1] if isinstance(estimator, tuple) else estimator
                if hasattr(est, 'feature_importances_'):
                    candidate = est
                    break
        elif hasattr(_model, 'feature_importances_'):
            candidate = _model

        if candidate is not None:
            return shap.TreeExplainer(candidate)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────
# INPUT VALIDATION
# ──────────────────────────────────────────────────────────────

def _validate_and_warn_metrics(metrics: dict) -> None:
    """
    Warns about missing or suspicious metric values.
    Never raises — prediction always continues with 0.0 defaults.
    """
    if not metrics:
        warnings.warn(
            "[DefectSense] predict_file() received empty metrics dict. "
            "All features default to 0.0 — prediction will be unreliable.",
            RuntimeWarning, stacklevel=3
        )
        return

    missing = [col for col in _feature_cols if col not in metrics]
    if missing:
        warnings.warn(
            f"[DefectSense] {len(missing)}/{len(_feature_cols)} features missing, "
            f"defaulting to 0.0: {missing[:8]}{'...' if len(missing) > 8 else ''}",
            RuntimeWarning, stacklevel=3
        )

    # Sanity check: these metrics should never be negative
    non_negative = ['loc', 'v(g)', 'branchCount', 'b', 'd', 'n', 'v', 'l', 'e']
    for col in non_negative:
        if col in metrics and float(metrics[col]) < 0:
            warnings.warn(
                f"[DefectSense] Metric '{col}' = {metrics[col]} is negative. "
                "This likely indicates a feature extraction error.",
                RuntimeWarning, stacklevel=3
            )


# ──────────────────────────────────────────────────────────────
# SHAP COMPUTATION  (real SHAP → improved proxy fallback)
# ──────────────────────────────────────────────────────────────

def _compute_shap_values(
    X_scaled: np.ndarray,
    X_unscaled: np.ndarray,
    raw_proba: float
) -> tuple[dict, bool]:
    """
    Returns (shap_dict, is_real_shap).
    Uses real SHAP TreeExplainer if shap is installed,
    otherwise falls back to a signed importance-deviation proxy.
    """

    # ── Real SHAP (best, install with: pip install shap) ─────
    if _shap_explainer is not None:
        try:
            shap_vals = _shap_explainer.shap_values(X_scaled)
            # Classifiers return [class0_shap, class1_shap]
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                vals = shap_vals[1][0]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals[0] if hasattr(shap_vals, '__len__') else shap_vals
            shap_dict = {col: float(vals[i]) for i, col in enumerate(_feature_cols)}
            return shap_dict, True
        except Exception:
            pass  # Fall through to proxy

    # ── Improved Proxy SHAP (fallback) ───────────────────────
    importances = _get_feature_importances()
    if len(importances) != len(_feature_cols):
        importances = np.resize(importances, len(_feature_cols))

    means = _scaler.mean_
    stds = _scaler.scale_
    deviations = (X_unscaled - means) / (stds + 1e-8)

    # Model confidence scales from -1 (very Clean) to +1 (very Defect-Prone)
    model_confidence = (raw_proba - 0.5) * 2
    shap_dict = {}
    for i, col in enumerate(_feature_cols):
        direction = 1.0 if deviations[i] > 0 else -1.0
        shap_dict[col] = float(
            importances[i] * abs(deviations[i]) * direction * model_confidence
        )
    return shap_dict, False


# ──────────────────────────────────────────────────────────────
# RISK SCORE  (0-100, human-readable for dashboard/demo)
# ──────────────────────────────────────────────────────────────

def _compute_risk_score(calibrated_proba: float) -> dict:
    """
    Converts probability to a 0-100 risk score with category + color.
    Makes the frontend dashboard much easier to understand during demos.
    """
    score = round(calibrated_proba * 100, 1)
    if score >= 75:
        return {'score': score, 'category': 'Critical', 'color': 'red'}
    if score >= 50:
        return {'score': score, 'category': 'High',     'color': 'orange'}
    if score >= 25:
        return {'score': score, 'category': 'Medium',   'color': 'yellow'}
    return     {'score': score, 'category': 'Low',      'color': 'green'}


# ──────────────────────────────────────────────────────────────
# MAIN PREDICTION ENTRY POINT
# ──────────────────────────────────────────────────────────────

def predict_file(metrics: dict, filename: str = '') -> dict:
    """
    Run ML prediction on extracted software metrics.

    Args:
        metrics  : dict of metric_name → float value
                   e.g. {'loc': 120, 'v(g)': 8, 'branchCount': 12, ...}
        filename : optional source file name for calibration context

    Returns:
        Full result dict:
          label            → 'Defect-Prone' | 'Clean'
          probability      → calibrated probability (0.0 – 1.0)
          raw_probability  → model's raw probability before calibration
          decision_threshold
          confidence_band  → 'very_high' | 'high' | 'medium' | 'low'
          risk_score       → {'score': 73.2, 'category': 'High', 'color': 'orange'}
          shap_values      → {feature: shap_value, ...}
          top_features     → top 5 features by |shap| impact
          shap_method      → 'real_shap' | 'proxy'
          calibration      → calibration metadata dict
          model_meta       → training metadata from train.py
    """
    _load_models()
    _validate_and_warn_metrics(metrics)

    # ── Build feature vector ──────────────────────────────────
    row = [float(metrics.get(col, 0.0)) for col in _feature_cols]
    X_unscaled = np.array(row, dtype=float)
    X_scaled = _scaler.transform(X_unscaled.reshape(1, -1))

    # ── Raw model output ──────────────────────────────────────
    raw_proba = float(_model.predict_proba(X_scaled)[0][1])

    # ── Decision threshold ────────────────────────────────────
    # train.py should save F1-optimal threshold in model_meta.
    # Fallback is 0.40 (better than 0.5 for imbalanced NASA data).
    decision_threshold = float((_model_meta or {}).get('decision_threshold', 0.40))

    # ── Calibration ───────────────────────────────────────────
    calibrated_proba, calibration = _apply_probability_calibration(
        raw_probability=raw_proba,
        threshold=decision_threshold,
        metrics=metrics,
        filename=filename or '',
    )

    label = "Defect-Prone" if calibrated_proba >= decision_threshold else "Clean"

    # ── SHAP values ───────────────────────────────────────────
    shap_values, is_real_shap = _compute_shap_values(X_scaled, X_unscaled, raw_proba)
    top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        # Core prediction
        'label':              label,
        'probability':        calibrated_proba,
        'raw_probability':    raw_proba,
        'decision_threshold': decision_threshold,

        # Confidence & risk
        'confidence_band':    _confidence_band(calibrated_proba, decision_threshold),
        'risk_score':         _compute_risk_score(calibrated_proba),

        # Explainability
        'shap_values':        shap_values,
        'top_features':       top_features,
        'shap_method':        'real_shap' if is_real_shap else 'proxy',

        # Metadata
        'calibration':        calibration,
        'model_meta':         _model_meta or {},
    }


# ──────────────────────────────────────────────────────────────
# UTILITY EXPORTS
# ──────────────────────────────────────────────────────────────

def get_model_meta() -> dict:
    try:
        _load_models()
        return _model_meta or {}
    except Exception:
        return {}


def is_model_loaded() -> bool:
    try:
        _load_models()
        return True
    except Exception:
        return False


def get_feature_columns() -> list:
    """Returns the exact feature columns the model was trained on."""
    try:
        _load_models()
        return list(_feature_cols)
    except Exception:
        return []


def predict_batch(metrics_list: list[dict], filenames: list[str] = None) -> list[dict]:
    """
    Bulk prediction for scanning multiple files at once.

    Args:
        metrics_list : list of metrics dicts
        filenames    : optional list of corresponding filenames

    Returns:
        List of prediction result dicts (same format as predict_file)
    """
    _load_models()
    if filenames is None:
        filenames = [''] * len(metrics_list)
    return [
        predict_file(m, f)
        for m, f in zip(metrics_list, filenames)
    ]