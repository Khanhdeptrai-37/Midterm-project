"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Automatic Modulation Classification (AMC) Across 8 Digital Modulation Schemes
Unified Interactive Dashboard: Classical ML (Random Forest, RBF-SVM) vs. Tier 2 Neural Network (3-Layer MLP)
"""

import os
import sys

# Delegate or run unified dashboard
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_app = os.path.join(ROOT_DIR, "app.py")

if os.path.exists(root_app):
    with open(root_app, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, root_app, 'exec'))
else:
    import app
