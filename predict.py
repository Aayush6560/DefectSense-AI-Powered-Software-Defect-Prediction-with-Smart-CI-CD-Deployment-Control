"""Compatibility layer exposing model inference functions.

This module keeps newer imports (`from ml.predict import ...`) working
while the core implementation remains in model.py.
"""

from model import get_model_meta, is_model_loaded, predict_file

__all__ = ["predict_file", "get_model_meta", "is_model_loaded"]
