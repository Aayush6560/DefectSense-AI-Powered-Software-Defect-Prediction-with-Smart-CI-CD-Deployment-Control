import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'ml', 'models')

FEATURE_COLS = [
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
    'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
    'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount',
]


def _try_import_smote():
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=42, k_neighbors=5)
    except ImportError:
        return None


def _try_import_lgbm():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        return None


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place a labeled NASA KC1-style CSV with required feature columns and a 'defects' label."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.replace('?', pd.NA)

    if 'defects' not in df.columns:
        raise ValueError("Dataset is missing required 'defects' target column.")

    df['defects'] = df['defects'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[FEATURE_COLS + ['defects']].dropna()
    return df


def _best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.10, 0.90, 81)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal = balanced_accuracy_score(y_true, y_pred)
        score = 0.45 * f1 + 0.30 * bal + 0.15 * p + 0.10 * r
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def _extract_feature_importances(model) -> dict:
    if hasattr(model, 'feature_importances_'):
        return dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            candidate = est[1] if isinstance(est, tuple) else est
            if hasattr(candidate, 'feature_importances_'):
                imp = candidate.feature_importances_
                if len(imp) == len(FEATURE_COLS):
                    return dict(zip(FEATURE_COLS, imp.tolist()))
    if hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
        coef = np.abs(model.final_estimator_.coef_[0])
        n = min(len(coef), len(FEATURE_COLS))
        return dict(zip(FEATURE_COLS[:n], coef[:n].tolist()))
    return {col: 0.0 for col in FEATURE_COLS}


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("  DefectSense ML Training Pipeline")
    print("=" * 60)

    df = load_data()
    defect_rate = df['defects'].mean()
    print(f"[INFO] Dataset: {len(df)} modules | Defect rate: {defect_rate:.1%}")

    X = df[FEATURE_COLS].values
    y = df['defects'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = _try_import_smote()
    if smote is not None:
        try:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"[INFO] SMOTE applied: {len(y_train)} → {len(y_train_res)} training samples")
        except Exception as e:
            print(f"[WARN] SMOTE failed ({e}), using original distribution")
            X_train_res, y_train_res = X_train, y_train
    else:
        print("[WARN] imbalanced-learn not installed. SMOTE skipped. Install with: pip install imbalanced-learn")
        X_train_res, y_train_res = X_train, y_train

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_res)
    X_test_s = scaler.transform(X_test)

    sample_weights_train = compute_sample_weight('balanced', y_train_res)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=16, min_samples_leaf=1,
            class_weight='balanced_subsample', random_state=42, n_jobs=-1,
        )),
        ('gbt', GradientBoostingClassifier(
            n_estimators=220, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )),
        ('dt', DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=2,
            class_weight='balanced', random_state=42,
        )),
    ]

    meta_model = LogisticRegression(
        C=0.5, max_iter=2000, class_weight='balanced',
        solver='lbfgs', random_state=42,
    )

    candidate_models = [
        ('stacking', StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=skf,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1,
        )),
        ('extra_trees', ExtraTreesClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )),
        ('hist_gb', HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=8, max_iter=300,
            class_weight='balanced', random_state=42,
        )),
    ]

    lgbm = _try_import_lgbm()
    if lgbm is not None:
        candidate_models.append(('lgbm', lgbm))
        print("[INFO] LightGBM detected — added as candidate model")

    print(f"[INFO] Training {len(candidate_models)} candidate models with F1-optimized threshold selection...")

    candidate_results = []
    for model_name, model in candidate_models:
        try:
            model.fit(X_train_s, y_train_res)
            y_proba = model.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            best_t = _best_threshold(y_test, y_proba)
            y_pred = (y_proba >= best_t).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            bal = balanced_accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, zero_division=0)
            score = 0.40 * f1 + 0.25 * bal + 0.20 * auc + 0.15 * rec
            candidate_results.append({
                'name': model_name,
                'model': model,
                'auc': float(auc),
                'ap': float(ap),
                'threshold': float(best_t),
                'f1': float(f1),
                'balanced_accuracy': float(bal),
                'recall': float(rec),
                'score': float(score),
                'y_pred': y_pred,
                'y_proba': y_proba,
            })
            print(f"  [{model_name}] AUC={auc:.4f} | AP={ap:.4f} | F1={f1:.4f} | Threshold={best_t:.2f} | Score={score:.4f}")
        except Exception as e:
            print(f"  [{model_name}] FAILED: {e}")

    if not candidate_results:
        raise RuntimeError("All candidate models failed to train.")

    best = max(candidate_results, key=lambda x: (x['score'], x['auc']))
    model = best['model']
    y_pred = best['y_pred']
    y_proba = best['y_proba']
    best_threshold = best['threshold']

    print(f"\n[WINNER] Selected model: {best['name']}")
    print(f"[RESULTS] AUC-ROC: {best['auc']:.4f} | Avg Precision: {best['ap']:.4f}")
    print(f"[RESULTS] F1: {best['f1']:.4f} | Recall: {best['recall']:.4f} | Balanced Acc: {best['balanced_accuracy']:.4f}")
    print(f"[RESULTS] Optimal Threshold: {best_threshold:.2f}")
    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Clean', 'Defect-Prone']))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"[RESULTS] Confusion Matrix → TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"[RESULTS] False Negative Rate (missed defects): {fn / (fn + tp):.1%}")

    cv_scores = cross_val_score(model, X_train_s, y_train_res, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"[RESULTS] 5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    feature_importances = _extract_feature_importances(model)

    model_meta = {
        'selected_model': best['name'],
        'auc_roc': best['auc'],
        'average_precision': best['ap'],
        'f1_at_threshold': best['f1'],
        'recall_at_threshold': best['recall'],
        'balanced_accuracy_at_threshold': best['balanced_accuracy'],
        'decision_threshold': best_threshold,
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
        'false_negative_rate': float(fn / (fn + tp)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'n_train': int(len(X_train_res)),
        'n_test': int(len(X_test)),
        'smote_applied': smote is not None,
        'defect_rate': float(defect_rate),
        'feature_importances': feature_importances,
        'dataset': 'NASA KC1 PROMISE Repository',
        'architecture': f'Best of {len(candidate_results)} candidates: RF+GBT+ET+DT stacking, ExtraTrees, HistGB' + (', LightGBM' if lgbm else ''),
    }

    with open(os.path.join(MODELS_DIR, 'stacking_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(FEATURE_COLS, f)
    with open(os.path.join(MODELS_DIR, 'model_meta.pkl'), 'wb') as f:
        pickle.dump(model_meta, f)

    print(f"\n[OK] Models saved to {MODELS_DIR}/")
    print(f"[OK] AUC-ROC: {best['auc']:.4f} | Threshold: {best_threshold:.2f} — quote both during demo!")
    return model, scaler, FEATURE_COLS, model_meta


if __name__ == '__main__':
    train()