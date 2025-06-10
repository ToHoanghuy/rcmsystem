"""
Hàm làm sạch dữ liệu JSON từ MongoDB nâng cao
Tích hợp vào hệ thống đề xuất để chuẩn hóa dữ liệu trả về
"""

import json
import re
import pandas as pd
import numpy as np

def clean_mongodb_json(data):
    """
    Làm sạch dữ liệu JSON từ MongoDB để đảm bảo tuân theo chuẩn JSON.
    Phiên bản nâng cao xử lý các trường hợp đặc biệt.
    
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
    """    # Kiểm tra nếu là list/array và xử lý ngay tại đây để tránh lỗi truth value
    if isinstance(data, list) or (hasattr(data, '__iter__') and not isinstance(data, (str, dict))):
        return [clean_mongodb_json(item) for item in data]
        
    # Xử lý giá trị NaN ngoài cùng
    try:
        # Xử lý các loại NaN khác nhau
        if data is None:
            return None
        if isinstance(data, float) and np.isnan(data):
            return None
        if isinstance(data, str) and data.lower() == "nan":
            return None
        if pd.isna(data):
            return None
    except (TypeError, ValueError):
        # Nếu có lỗi khi kiểm tra NaN (ví dụ với mảng), bỏ qua và tiếp tục
        pass
        
    if isinstance(data, dict):
        result = {}        
        for key, value in data.items():
            # Xử lý giá trị NaN một cách an toàn
            try:
                if value is None:
                    result[key] = None
                    continue
                if isinstance(value, float) and np.isnan(value):
                    result[key] = None
                    continue
                if isinstance(value, str) and value.lower() == "nan":
                    result[key] = None
                    continue
                if pd.isna(value):
                    result[key] = None
                    continue
            except (TypeError, ValueError):
                # Nếu có lỗi khi kiểm tra NaN, xử lý riêng cho mảng
                if isinstance(value, (list, pd.Series, np.ndarray)):
                    result[key] = clean_mongodb_json(value)
                    continue
                
            # Xử lý chuỗi JSON lồng nhau
            if isinstance(value, str) and value.startswith('{') and (key == 'category' or key == 'caategory' or 'category' in key.lower()):
                try:
                    result[key] = json.loads(value.strip())
                    continue
                except Exception as e:
                    print(f"Error parsing JSON string in key {key}: {str(e)}")
            
            # Xử lý chuỗi mảng đặc biệt (như trường image)
            if isinstance(value, str) and (value.startswith('[{') or value.startswith('[') or value.startswith('{')):
                try:
                    # Bước 1: Thay thế các tham chiếu ObjectId
                    cleaned_str = re.sub(r"ObjectId\(['\"]([a-f0-9]+)['\"]\)", r'"\1"', value)
                    
                    # Bước 2: Thêm dấu nháy kép cho các key thiếu
                    cleaned_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', cleaned_str)
                    
                    # Bước 3: Thay thế dấu nháy đơn bằng nháy kép (nếu là giá trị chuỗi)
                    cleaned_str = cleaned_str.replace("'", '"')
                    
                    # Bước 4: Tải và xử lý đệ quy
                    parsed_value = json.loads(cleaned_str)
                    result[key] = clean_mongodb_json(parsed_value)
                except Exception as e:
                    # Xử lý đặc biệt cho trường image
                    if key == 'image':
                        # Trích xuất các URL trong chuỗi
                        try:
                            # Tìm tất cả URL trong chuỗi
                            url_pattern = r"'url':\s*'([^']+)'"
                            # Hoặc dạng "url": "..."
                            if '"url"' in value:
                                url_pattern = r'"url":\s*"([^"]+)"'
                                
                            urls = re.findall(url_pattern, value)
                            if urls:
                                result[key] = [{"url": url} for url in urls]
                            else:
                                # Nếu không tìm thấy URL, trả về mảng rỗng
                                result[key] = []
                        except:
                            # Nếu không thể trích xuất, trả về mảng rỗng
                            result[key] = []
                    else:
                        # Đối với các trường khác, giữ nguyên giá trị
                        result[key] = value
                        
            # Xử lý đệ quy cho mảng và đối tượng
            elif isinstance(value, dict):
                result[key] = clean_mongodb_json(value)
            elif isinstance(value, list):
                result[key] = [clean_mongodb_json(item) for item in value]
            else:
                # Chuyển các kiểu numpy về kiểu Python chuẩn (dù không phải NaN)
                if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    result[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                    result[key] = float(value)
                elif isinstance(value, np.bool_):
                    result[key] = bool(value)
                else:
                    result[key] = value
                
        return result    
    elif isinstance(data, list):
        return [clean_mongodb_json(item) for item in data]
    elif isinstance(data, pd.Series):
        # Chuyển Pandas Series thành list và xử lý
        return [clean_mongodb_json(item) for item in data.tolist()]
    elif isinstance(data, np.ndarray):
        # Chuyển NumPy array thành list và xử lý
        return [clean_mongodb_json(item) for item in data.tolist()]
    else:
        # Chuyển các kiểu numpy về kiểu Python chuẩn
        if isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        if isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        if isinstance(data, np.bool_):
            return bool(data)
        return data
