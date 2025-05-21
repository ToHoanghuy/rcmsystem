"""
Module cập nhật cho RealtimeRecommender để hỗ trợ event tracking từ frontend
"""

# Định nghĩa phương thức mới cho RealtimeRecommender
def add_event(self, user_id, location_id, event_type, data=None, timestamp=None):
    """
    Ghi nhận sự kiện người dùng mới
    
    Parameters:
    -----------
    user_id: str
        ID của người dùng
    location_id: str
        ID của địa điểm
    event_type: str
        Loại sự kiện ('view', 'click', 'book', 'rate', 'search', 'favorite')
    data: dict
        Dữ liệu bổ sung kèm theo sự kiện
    timestamp: datetime, optional
        Thời gian xảy ra sự kiện, mặc định là thời gian hiện tại
    """
    from datetime import datetime
    import logging
    logger = logging.getLogger(__name__)
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Make sure user_id and location_id are strings to prevent int() conversion errors
    user_id = str(user_id) if user_id is not None else None
    location_id = str(location_id) if location_id is not None else None
    
    # Đảm bảo data là một dict
    data = data or {}
        
    # Đảm bảo rằng recent_events được khởi tạo
    if not hasattr(self, 'recent_events') or self.recent_events is None:
        self.recent_events = {}
    
    # Khởi tạo danh sách sự kiện cho người dùng nếu chưa có
    if user_id not in self.recent_events:
        self.recent_events[user_id] = []
    
    # Thêm sự kiện mới
    self.recent_events[user_id].append({
        'location_id': location_id,
        'event_type': event_type,
        'timestamp': timestamp,
        'data': data
    })
    
    # Giới hạn số lượng sự kiện lưu trữ
    if len(self.recent_events[user_id]) > self.max_events_per_user:
        self.recent_events[user_id] = self.recent_events[user_id][-self.max_events_per_user:]
        
    # Cập nhật ma trận đồng xuất hiện cho các loại sự kiện phù hợp
    if event_type in ['view', 'click', 'book']:
        # Gọi phương thức cập nhật ma trận đồng xuất hiện với tên sản phẩm đúng
        self._update_cooccurrence_matrices(user_id, location_id, event_type)
        
    logger.info(f"Added {event_type} event for user {user_id} on location {location_id}")
    return True

def add_rating(self, user_id, location_id, rating, timestamp=None):
    """
    Thêm hoặc cập nhật rating từ người dùng
    
    Parameters:
    -----------
    user_id: str
        ID của người dùng
    location_id: str
        ID của địa điểm
    rating: float
        Điểm đánh giá (thường từ 1-5)
    timestamp: datetime, optional
        Thời gian đánh giá, mặc định là thời gian hiện tại
    """
    from datetime import datetime
    import os
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Khởi tạo dict ratings cho người dùng nếu chưa có
    if user_id not in self.recent_ratings:
        self.recent_ratings[user_id] = {}
    
    # Cập nhật rating
    self.recent_ratings[user_id][location_id] = rating
    
    # Lưu vào file CSV
    try:
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(self.ratings_path):
            ratings_df = pd.read_csv(self.ratings_path)
            
            # Kiểm tra xem rating này đã tồn tại chưa
            mask = (ratings_df['user_id'] == user_id) & (ratings_df['location_id'] == location_id)
            if mask.any():
                # Cập nhật rating và timestamp nếu đã tồn tại
                ratings_df.loc[mask, 'rating'] = rating
                ratings_df.loc[mask, 'timestamp'] = timestamp.isoformat()
            else:
                # Thêm rating mới nếu chưa tồn tại
                new_rating = pd.DataFrame({
                    'user_id': [user_id],
                    'location_id': [location_id],
                    'rating': [rating],
                    'timestamp': [timestamp.isoformat()]
                })
                ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
                
            # Lưu lại file
            ratings_df.to_csv(self.ratings_path, index=False)
            logger.info(f"Updated rating for user {user_id} on location {location_id}: {rating}")
        else:
            # Tạo file mới với rating đầu tiên
            ratings_df = pd.DataFrame({
                'user_id': [user_id],
                'location_id': [location_id],
                'rating': [rating],
                'timestamp': [timestamp.isoformat()]
            })
            
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(self.ratings_path), exist_ok=True)
            
            # Lưu file
            ratings_df.to_csv(self.ratings_path, index=False)
            logger.info(f"Created new ratings file with first rating from user {user_id}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving rating to CSV: {str(e)}")
        return False

# Define additional patch methods directly here
def _update_cooccurrence_matrices(self, user_id, current_location_id, event_type):
    """
    Cập nhật ma trận đồng xuất hiện dựa trên sự kiện mới
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Safety check for recent_events
    if not hasattr(self, 'recent_events') or self.recent_events is None:
        self.recent_events = {}
        logger.warning("recent_events was not initialized, creating it now")
        return
        
    # Safety check for coview_matrix
    if not hasattr(self, 'coview_matrix') or self.coview_matrix is None:
        self.coview_matrix = {}
        logger.warning("coview_matrix was not initialized, creating it now")
        
    # Safety check for copurchase_matrix
    if not hasattr(self, 'copurchase_matrix') or self.copurchase_matrix is None:
        self.copurchase_matrix = {}
        logger.warning("copurchase_matrix was not initialized, creating it now")
        
    if user_id not in self.recent_events or len(self.recent_events[user_id]) < 2:
        return
        
    # Lấy các địa điểm gần đây người dùng đã tương tác
    recent_locations = [event['location_id'] for event in self.recent_events[user_id][:-1]]
      
    # Cập nhật ma trận tương quan
    if event_type == 'view':
        for prev_location_id in recent_locations:
            if prev_location_id != current_location_id:
                # Cập nhật co-view matrix
                key = self._get_location_pair_key(prev_location_id, current_location_id)
                self.coview_matrix[key] = self.coview_matrix.get(key, 0) + 1
    
    elif event_type == 'book':  # 'book' thay cho 'purchase' cho ứng dụng du lịch
        # Lấy các địa điểm xem gần đây trước khi đặt
        for event in self.recent_events[user_id]:
            if event['event_type'] == 'view' and event['location_id'] != current_location_id:
                # Cập nhật co-purchase matrix
                key = self._get_location_pair_key(event['location_id'], current_location_id)
                self.copurchase_matrix[key] = self.copurchase_matrix.get(key, 0) + 1

def _get_location_pair_key(self, location_id1, location_id2):
    """
    Tạo khóa duy nhất cho cặp địa điểm (không phân biệt thứ tự)
    """
    # Sắp xếp để đảm bảo tính nhất quán (không phân biệt thứ tự)
    ids = sorted([str(location_id1), str(location_id2)])
    return f"{ids[0]}_{ids[1]}"

# Thêm phương thức để patch vào RealtimeRecommender
def patch_realtime_recommender():
    """
    Thêm các phương thức mới vào class RealtimeRecommender
    """
    from recommenders.realtime import RealtimeRecommender
    
    # Patching instance methods by replacing the existing methods with new ones
    RealtimeRecommender.add_event = add_event
    RealtimeRecommender.add_rating = add_rating
    RealtimeRecommender._update_cooccurrence_matrices = _update_cooccurrence_matrices
    RealtimeRecommender._get_location_pair_key = _get_location_pair_key
    
    # Also make sure the recent_events attribute exists in the class initialization
    original_init = RealtimeRecommender.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Always initialize these attributes to ensure they exist
        self.recent_events = getattr(self, 'recent_events', {})
        self.recent_ratings = getattr(self, 'recent_ratings', {})
        self.max_events_per_user = getattr(self, 'max_events_per_user', 20)
        self.coview_matrix = getattr(self, 'coview_matrix', {})
        self.copurchase_matrix = getattr(self, 'copurchase_matrix', {})
    
    RealtimeRecommender.__init__ = new_init
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info("RealtimeRecommender successfully patched with add_event, add_rating, and cooccurrence methods")
