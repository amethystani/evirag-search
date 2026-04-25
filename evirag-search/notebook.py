#!/usr/bin/env python3
"""
EVIRAG Pipeline for Kaggle — Complete end-to-end execution.
Push and run via:
  kaggle kernels push -p evirag-search/
"""

import subprocess
import sys
import os

# Run the full pipeline
result = subprocess.run([sys.executable, "kaggle_runner.py"], cwd="/kaggle/input/evirag-search")
sys.exit(result.returncode)
