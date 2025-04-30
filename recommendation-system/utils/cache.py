import hashlib
import time
import json
from functools import lru_cache

class RecommendationCache:
    """
    Class quản lý cache cho kết quả gợi ý
    
    Attributes:
    -----------
    cache_data: dict
        Dictionary lưu trữ kết quả cache
    cache_expiry: int
        Thời gian hết hạn của cache (tính bằng giây)
    max_size: int
        Kích thước tối đa của cache
    """
    
    def __init__(self, cache_expiry=3600, max_size=1000):
        """
        Khởi tạo cache với thời gian hết hạn và kích thước tối đa
        
        Parameters:
        -----------
        cache_expiry: int
            Thời gian hết hạn của cache (tính bằng giây)
        max_size: int
            Kích thước tối đa của cache
        """
        self.cache_data = {}
        self.cache_expiry = cache_expiry
        self.max_size = max_size
    
    def _get_cache_key(self, user_id, case, product_id=None, method=None):
        """
        Tạo khóa cache duy nhất từ các tham số
        
        Parameters:
        -----------
        user_id: int
            ID của người dùng
        case: str
            Loại gợi ý (collaborative, content_based, hybrid, behavior_based)
        product_id: int, optional
            ID của sản phẩm (nếu có)
        method: str, optional
            Phương pháp sử dụng cho hybrid (adaptive, switching)
            
        Returns:
        --------
        str
            Khóa cache duy nhất
        """
        key = f"user_{user_id}_case_{case}"
        if product_id is not None:
            key += f"_product_{product_id}"
        if method is not None:
            key += f"_method_{method}"
        
        # Tạo hash MD5 từ khóa
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_cached_recommendation(self, user_id, case, product_id=None, method=None):
        """
        Lấy kết quả gợi ý từ cache (nếu có)
        
        Parameters:
        -----------
        user_id: int
            ID của người dùng
        case: str
            Loại gợi ý (collaborative, content_based, hybrid, behavior_based)
        product_id: int, optional
            ID của sản phẩm (nếu có)
        method: str, optional
            Phương pháp sử dụng cho hybrid (adaptive, switching)
            
        Returns:
        --------
        tuple or None
            (recommendations, timestamp) nếu có trong cache và chưa hết hạn, None nếu không có
        """
        cache_key = self._get_cache_key(user_id, case, product_id, method)
        
        # Kiểm tra xem khóa có trong cache không
        if cache_key in self.cache_data:
            timestamp, recommendations = self.cache_data[cache_key]
            
            # Kiểm tra xem cache có hết hạn chưa
            if time.time() - timestamp < self.cache_expiry:
                return recommendations
            else:
                # Xóa cache đã hết hạn
                del self.cache_data[cache_key]
        
        return None
    
    def cache_recommendation(self, user_id, case, recommendations, product_id=None, method=None):
        """
        Lưu trữ kết quả gợi ý vào cache
        
        Parameters:
        -----------
        user_id: int
            ID của người dùng
        case: str
            Loại gợi ý (collaborative, content_based, hybrid, behavior_based)
        recommendations: object
            Kết quả gợi ý
        product_id: int, optional
            ID của sản phẩm (nếu có)
        method: str, optional
            Phương pháp sử dụng cho hybrid (adaptive, switching)
            
        Returns:
        --------
        None
        """
        cache_key = self._get_cache_key(user_id, case, product_id, method)
        
        # Nếu cache đã đầy, xóa một mục ngẫu nhiên
        if len(self.cache_data) >= self.max_size:
            # Xóa mục cũ nhất
            oldest_key = min(self.cache_data.keys(), key=lambda k: self.cache_data[k][0])
            del self.cache_data[oldest_key]
        
        # Lưu trữ kết quả vào cache
        self.cache_data[cache_key] = (time.time(), recommendations)
    
    def clear_cache(self):
        """
        Xóa tất cả các mục trong cache
        
        Returns:
        --------
        None
        """
        self.cache_data = {}
    
    def clear_expired_cache(self):
        """
        Xóa các mục đã hết hạn trong cache
        
        Returns:
        --------
        int
            Số lượng mục đã xóa
        """
        current_time = time.time()
        expired_keys = [key for key, (timestamp, _) in self.cache_data.items()
                        if current_time - timestamp >= self.cache_expiry]
        
        # Xóa các khóa đã hết hạn
        for key in expired_keys:
            del self.cache_data[key]
        
        return len(expired_keys)
    
    def get_cache_stats(self):
        """
        Lấy thông tin thống kê về cache
        
        Returns:
        --------
        dict
            Thông tin thống kê
        """
        current_time = time.time()
        active_items = 0
        expired_items = 0
        
        for timestamp, _ in self.cache_data.values():
            if current_time - timestamp < self.cache_expiry:
                active_items += 1
            else:
                expired_items += 1
        
        return {
            'total_items': len(self.cache_data),
            'active_items': active_items,
            'expired_items': expired_items,
            'max_size': self.max_size,
            'cache_expiry': self.cache_expiry
        }


# Khởi tạo cache toàn cục
recommendation_cache = RecommendationCache()

@lru_cache(maxsize=128)
def get_cached_prediction(user_id, product_id, model_name):
    """
    Wrapper cho hàm dự đoán sử dụng decorator lru_cache
    
    Parameters:
    -----------
    user_id: int
        ID của người dùng
    product_id: int
        ID của sản phẩm
    model_name: str
        Tên mô hình
        
    Returns:
    --------
    float
        Điểm đánh giá dự đoán
    """
    # Đây là một hàm giả định, cần được thay thế bằng hàm dự đoán thực tế
    return 0.0