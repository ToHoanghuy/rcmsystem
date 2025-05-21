"""
Additional methods for patching the RealtimeRecommender class

This module simply re-exports the required functions from realtime_patch.py
to maintain backward compatibility with any code that imports them from here.
"""

import logging
logger = logging.getLogger(__name__)

# Import the required functions from realtime_patch.py
from utils.realtime_patch import _update_cooccurrence_matrices, _get_location_pair_key

# Export them from this module too
__all__ = ['_update_cooccurrence_matrices', '_get_location_pair_key']
