"""
Utilities for JSON serialization
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, date

class JSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles NumPy types and other special objects
    """
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Let the default encoder handle the rest
        return super().default(obj)

def dumps(obj, **kwargs):
    """
    Serialize obj to JSON formatted string with custom encoder
    """
    return json.dumps(obj, cls=JSONEncoder, **kwargs)

def convert_to_json_serializable(obj):
    """
    Recursively converts an object to be JSON serializable by converting
    problematic types (like NumPy types) to standard Python types
    
    Parameters:
    -----------
    obj : object
        The object to convert
        
    Returns:
    --------
    object
        A JSON-serializable version of the object
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj
