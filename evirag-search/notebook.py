#!/usr/bin/env python3
"""
EVIRAG ULTRA-FAST Pipeline for Kaggle (4 hours, 500k papers, T4x2)
Push and run via:
  kaggle kernels push -p evirag-search/ -u
"""

import subprocess
import sys
import os

# Run the FAST pipeline (optimized for 4-hour limit)
result = subprocess.run([sys.executable, "fast_pipeline.py"], cwd="/kaggle/input/evirag-search")
sys.exit(result.returncode)
