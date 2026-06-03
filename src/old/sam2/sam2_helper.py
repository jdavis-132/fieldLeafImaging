"""
Convenient wrapper for SAM2TinyPredictor
Import this instead of directly importing sam2_tiny
"""

import sys
import os

# Add sam2 directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_sam2_path = os.path.join(_current_dir, 'src', 'sam2')
if _sam2_path not in sys.path:
    sys.path.insert(0, _sam2_path)

# Now import SAM2TinyPredictor
from sam2_tiny import SAM2TinyPredictor

__all__ = ['SAM2TinyPredictor']
