"""
Test script to verify JSON serialization of NumPy types
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.json_utils import dumps, convert_to_json_serializable

def test_json_serialization():
    """
    Test JSON serialization with NumPy types
    """
    print("Testing JSON serialization with NumPy types...")
    
    # Create test data with NumPy types
    test_data = {
        "int64": np.int64(123456789),
        "float64": np.float64(123.456),
        "array": np.array([1, 2, 3, 4, 5]),
        "bool": np.bool_(True),
        "datetime": datetime.now(),
        "nested": {
            "int32": np.int32(42),
            "float32": np.float32(3.14)
        },
        "list_with_numpy": [np.int64(1), np.int64(2), np.int64(3)],
        "dataframe_values": pd.DataFrame({
            'id': [np.int64(1), np.int64(2), np.int64(3)],
            'value': [np.float64(10.5), np.float64(20.5), np.float64(30.5)]
        }).to_dict('records')
    }
    
    # Try with standard json dumps (should fail)
    try:
        json_str = json.dumps(test_data)
        print("Standard json.dumps succeeded (unexpected)")
    except TypeError as e:
        print(f"Standard json.dumps failed (expected): {e}")
    
    # Try with our custom JSON encoder
    try:
        json_str = dumps(test_data)
        print("Custom dumps succeeded (expected)")
        print(f"Serialized data: {json_str[:100]}...")  # Print first 100 chars
    except Exception as e:
        print(f"Custom dumps failed (unexpected): {e}")
    
    # Try with convert function
    try:
        converted_data = convert_to_json_serializable(test_data)
        json_str = json.dumps(converted_data)
        print("convert_to_json_serializable succeeded (expected)")
        print(f"Converted and serialized data: {json_str[:100]}...")  # Print first 100 chars
    except Exception as e:
        print(f"convert_to_json_serializable failed (unexpected): {e}")

if __name__ == "__main__":
    test_json_serialization()
