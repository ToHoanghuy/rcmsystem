"""
Utilities for JSON serialization
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
import re

class JSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles NumPy types and other special objects
    """
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            # Check for NaN and Infinity
            if np.isnan(obj) or obj is float('nan'):
                return None
            if np.isinf(obj):
                return str(obj)  # Convert infinity to string
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
            
        # Handle NaN values directly
        if pd.isna(obj) or obj is float('nan'):
            return None

        # Let the default encoder handle the rest
        return super().default(obj)

def dumps(obj, **kwargs):
    """
    Serialize obj to JSON formatted string with custom encoder
    """
    # Set ensure_ascii=False để giữ nguyên các ký tự Unicode (tiếng Việt)
    if 'ensure_ascii' not in kwargs:
        kwargs['ensure_ascii'] = False
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
    elif pd.isna(obj) or obj is float('nan') or obj == "NaN" or str(obj) == "nan":
        return None
    elif isinstance(obj, str):
        # Đảm bảo chuỗi Unicode được xử lý đúng
        try:
            # Nếu là chuỗi không hợp lệ, chuyển đổi an toàn
            return obj.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            return obj
    # Xử lý các kiểu dữ liệu không được hỗ trợ khác
    elif hasattr(obj, 'to_dict'):
        # Xử lý đối tượng có phương thức to_dict (như Pandas Series)
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):
        # Xử lý đối tượng có phương thức tolist (như numpy arrays)
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):
        # Xử lý các đối tượng ngày/giờ
        return obj.isoformat()
    else:
        return obj

def clean_json_data(data):
    """
    Làm sạch dữ liệu JSON từ MongoDB để đảm bảo tuân theo chuẩn JSON.
    
    Giải quyết các vấn đề phổ biến:
    - Chuyển đổi giá trị NaN thành null
    - Xử lý chuỗi JSON lồng nhau (như field 'category') 
    - Xử lý chuỗi mảng đặc biệt (như field 'image')
    - Xử lý ObjectId và các kiểu dữ liệu MongoDB khác
    - Đảm bảo mã hóa Unicode đúng cho tiếng Việt
    
    Parameters:
    -----------
    data : object
        Dữ liệu cần làm sạch (dict, list hoặc kiểu dữ liệu khác)
        
    Returns:
    --------
    object
        Dữ liệu đã được làm sạch tuân theo chuẩn JSON
    """
    # Đặc biệt xử lý NaN có thể được truyền trực tiếp
    if pd.isna(data) or data is float('nan'):
        return None

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Xử lý giá trị NaN và NaN đặc biệt
            if pd.isna(value) or value == float('nan') or value == "NaN" or value == "NaN" or value is float('nan'):
                result[key] = None
                continue
            
            # Xử lý chuỗi JSON lồng nhau
            if isinstance(value, str) and value.startswith('{') and (key == 'category' or key == 'caategory' or 'category' in key.lower()):
                try:
                    result[key] = json.loads(value.strip())
                    continue
                except Exception as e:
                    print(f"Error parsing JSON string in key {key}: {str(e)}")
                    # Tiếp tục với mã dưới để thử phương pháp khác
            
            # Xử lý chuỗi mảng/đối tượng đặc biệt (như trường image)
            if isinstance(value, str) and (value.startswith('[{') or value.startswith('[') or value.startswith('{')):
                try:
                    # Làm sạch chuỗi trước khi phân tích
                    # 1. Thay thế các tham chiếu ObjectId
                    cleaned_str = re.sub(r"ObjectId\(['\"]([a-f0-9]+)['\"]\)", r'"\1"', value)
                    # 2. Thêm dấu nháy kép cho các key thiếu
                    cleaned_str = re.sub(r"([']*(\w+)[']*\s*:)", r'"\2":', cleaned_str)
                    # 3. Thay thế dấu nháy đơn bằng nháy kép cho giá trị chuỗi
                    cleaned_str = cleaned_str.replace("'", '"')
                    # 4. Đảm bảo đóng mở ngoặc đúng
                    if cleaned_str.count('[') != cleaned_str.count(']') or cleaned_str.count('{') != cleaned_str.count('}'):
                        raise ValueError(f"Unbalanced brackets in JSON string: {cleaned_str}")
                    
                    # Tải và xử lý đệ quy
                    parsed_value = json.loads(cleaned_str)
                    result[key] = clean_json_data(parsed_value)
                except Exception as e:
                    # Nếu không thể phân tích JSON, thử tạo một mảng/đối tượng từ chuỗi này
                    print(f"Không thể phân tích chuỗi JSON cho {key}: {str(e)}")
                    if key == 'image' and value.startswith('[{') and "url" in value and "'" in value:
                        try:
                            # Trích xuất url từ chuỗi (trường hợp đặc biệt cho trường image)
                            urls = re.findall(r"'url':\s*'([^']+)'", value)
                            if urls:
                                result[key] = [{"url": url} for url in urls]
                                continue
                        except:
                            pass
                    result[key] = value
            # Xử lý đệ quy cho mảng và đối tượng
            elif isinstance(value, dict):
                result[key] = clean_json_data(value)
            elif isinstance(value, list):
                result[key] = [clean_json_data(item) for item in value]
            else:
                # Chuyển đổi giá trị thông thường
                result[key] = convert_to_json_serializable(value)
        return result
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    else:
        # Chuyển đổi giá trị cơ bản
        return convert_to_json_serializable(data)
