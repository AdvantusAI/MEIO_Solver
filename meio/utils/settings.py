"""
Settings utility functions for the MEIO system.
This module mainly serves as a compatibility layer for code that used the old settings approach.
"""
import os
from meio.utils.path_manager import paths

# For backward compatibility - use paths from path_manager
RESULTS_DIR = paths.RESULTS_DIR

# Make sure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True) 