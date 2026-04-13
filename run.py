import os
from app import create_app

app = create_app()

def _preload_model():
    try:
        from ml.predict import is_model_loaded
        if is_model_loaded():
            print("[OK] ML model loaded and ready")
        else:
            print("[WARN] ML model not loaded — run: python ml/train.py")
    except Exception as e:
        print(f"[WARN] Model preload check failed: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    _preload_model()

    print(f"""
╔══════════════════════════════════════════╗
║       DefectSense MLOps Platform         ║
║  AI-Powered Software Defect Detection    ║
╠══════════════════════════════════════════╣
║  URL:    http://localhost:{port:<5}          ║
║  Health: http://localhost:{port}/health   ║
║  Debug:  {str(debug):<33} ║
╚══════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=port, debug=debug)