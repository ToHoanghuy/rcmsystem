"""
Simple test script for JSON serialization
"""
import numpy as np
import json
from utils.json_utils import JSONEncoder, convert_to_json_serializable

# Create test data with NumPy types
test_data = {
    "int64": np.int64(123456789),
    "float64": np.float64(123.456),
    "array": np.array([1, 2, 3, 4, 5])
}

print("Testing JSON serialization with NumPy types...")

# Try standard serialization (should fail)
try:
    json_str = json.dumps(test_data)
    print("Standard json.dumps succeeded (unexpected)")
except TypeError as e:
    print(f"Standard json.dumps failed (expected): {e}")

# Try with our custom JSON encoder
try:
    json_str = json.dumps(test_data, cls=JSONEncoder)
    print("Custom encoder succeeded (expected)")
    print(f"Serialized data: {json_str}")
except Exception as e:
    print(f"Custom encoder failed (unexpected): {e}")

# Try with convert function
try:
    converted_data = convert_to_json_serializable(test_data)
    json_str = json.dumps(converted_data)
    print("convert_to_json_serializable succeeded (expected)")
    print(f"Converted and serialized data: {json_str}")
except Exception as e:
    print(f"convert_to_json_serializable failed (unexpected): {e}")
