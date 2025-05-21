import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from utils.json_utils import convert_to_json_serializable
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimeRecommender:
    """
    Lớp xử lý gợi ý theo thời gian thực dựa trên sự kiện người dùng
    """
    def __init__(self):
        # Lưu trữ sự kiện gần đây của người dùng (user_id -> [events])
        self.recent_events = {}
        # Lưu trữ ratings gần đây (user_id -> {location_id: rating})
        self.recent_ratings = {}
        # Lưu trữ mối tương quan giữa địa điểm xem cùng nhau (co-view matrix)
        self.coview_matrix = {}
        # Lưu trữ mối tương quan giữa địa điểm đặt phòng cùng nhau (co-book matrix)
        self.cobook_matrix = {}
        # Maximum số lượng sự kiện lưu trữ cho mỗi người dùng
        self.max_events_per_user = 20
        # Thời gian hết hạn cho sự kiện (tính bằng giây)
        self.event_expiry = 3600 * 24  # 24 giờ
        # Đường dẫn đến file ratings
        self.ratings_path = os.path.join(Config.RAW_DATA_DIR, 'location_ratings.csv')
    
    def record_event(self, user_id, location_id, event_type, timestamp=None):
        """
        Ghi nhận sự kiện người dùng mới (phương thức cũ để tương thích)
        
        Parameters:
        -----------
        user_id: str or int
            ID của người dùng
        location_id: str or int
            ID của địa điểm
        event_type: str
            Loại sự kiện ('view', 'click', 'book', 'rate', 'search', 'favorite')
        timestamp: datetime, optional
            Thời gian xảy ra sự kiện, mặc định là thời gian hiện tại
            
        Returns:
        --------
        bool
            True nếu ghi nhận thành công, False nếu có lỗi
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Đảm bảo ID là string
            user_id = str(user_id)
            location_id = str(location_id)
                
            # Khởi tạo danh sách sự kiện cho người dùng nếu chưa có
            if user_id not in self.recent_events:
                self.recent_events[user_id] = []
            
            # Thêm sự kiện mới
            self.recent_events[user_id].append({
                'location_id': location_id,
                'event_type': event_type,
                'timestamp': timestamp,
                'data': {}
            })
            
            # Giới hạn số lượng sự kiện lưu trữ
            if len(self.recent_events[user_id]) > self.max_events_per_user:
                self.recent_events[user_id] = self.recent_events[user_id][-self.max_events_per_user:]
            
            # Cập nhật ma trận đồng xuất hiện
            self._update_cooccurrence_matrices(user_id, location_id, event_type)
            
            logger.info(f"Recorded event for user {user_id}: {event_type} on location {location_id}")
            return True
        except Exception as e:
            logger.error(f"Error recording event: {str(e)}")
            return False
    
    def add_event(self, user_id, location_id, event_type, data=None, timestamp=None):
        """
        Ghi nhận sự kiện người dùng mới (phương thức mới từ frontend)
        
        Parameters:
        -----------
        user_id: str or int
            ID của người dùng
        location_id: str or int
            ID của địa điểm
        event_type: str
            Loại sự kiện ('view', 'click', 'book', 'rate', 'search', 'favorite')
        data: dict
            Dữ liệu bổ sung kèm theo sự kiện
        timestamp: datetime, optional
            Thời gian xảy ra sự kiện, mặc định là thời gian hiện tại
            
        Returns:
        --------
        bool
            True nếu ghi nhận thành công, False nếu có lỗi
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Đảm bảo ID là string
            user_id = str(user_id)
            location_id = str(location_id)
            
            # Khởi tạo danh sách sự kiện cho người dùng nếu chưa có
            if user_id not in self.recent_events:
                self.recent_events[user_id] = []
            
            # Thêm sự kiện mới
            self.recent_events[user_id].append({
                'location_id': location_id,
                'event_type': event_type,
                'timestamp': timestamp,
                'data': data or {}
            })
            
            # Giới hạn số lượng sự kiện lưu trữ
            if len(self.recent_events[user_id]) > self.max_events_per_user:
                self.recent_events[user_id] = self.recent_events[user_id][-self.max_events_per_user:]
            
            # Cập nhật ma trận đồng xuất hiện
            self._update_cooccurrence_matrices(user_id, location_id, event_type)
            
            logger.info(f"Added event for user {user_id}: {event_type} on location {location_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding event: {str(e)}")
            return False
    
    def add_rating(self, user_id, location_id, rating, timestamp=None):
        """
        Thêm hoặc cập nhật rating từ người dùng
        
        Parameters:
        -----------
        user_id: str or int
            ID của người dùng
        location_id: str or int
            ID của địa điểm
        rating: float
            Điểm đánh giá (thường từ 1-5)
        timestamp: datetime, optional
            Thời gian đánh giá, mặc định là thời gian hiện tại
            
        Returns:
        --------
        bool
            True nếu thành công, False nếu có lỗi
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Đảm bảo ID là string
            user_id = str(user_id)
            location_id = str(location_id)
            
            # Khởi tạo dict ratings cho người dùng nếu chưa có
            if user_id not in self.recent_ratings:
                self.recent_ratings[user_id] = {}
            
            # Cập nhật rating
            self.recent_ratings[user_id][location_id] = float(rating)
            
            # Lưu vào file CSV
            try:
                # Tạo thư mục cha của file nếu chưa tồn tại
                os.makedirs(os.path.dirname(self.ratings_path), exist_ok=True)
                
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
                else:
                    # Tạo DataFrame mới nếu file chưa tồn tại
                    ratings_df = pd.DataFrame({
                        'user_id': [user_id],
                        'location_id': [location_id],
                        'rating': [rating],
                        'timestamp': [timestamp.isoformat()]
                    })
                
                # Lưu vào file CSV
                ratings_df.to_csv(self.ratings_path, index=False)
                logger.info(f"Saved rating {rating} for user {user_id} on location {location_id}")
                
            except Exception as e:
                logger.error(f"Error saving rating to CSV: {str(e)}")
                
            return True
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            return False
        
    def _update_cooccurrence_matrices(self, user_id, current_location_id, event_type):
        """
        Cập nhật ma trận đồng xuất hiện dựa trên sự kiện mới
        
        Parameters:
        -----------
        user_id: str
            ID của người dùng
        current_location_id: str
            ID của địa điểm hiện tại
        event_type: str
            Loại sự kiện
        """
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
                    # Cập nhật co-book matrix
                    key = self._get_location_pair_key(event['location_id'], current_location_id)
                    self.cobook_matrix[key] = self.cobook_matrix.get(key, 0) + 1

    def _get_location_pair_key(self, location_id1, location_id2):
        """
        Tạo khóa duy nhất cho cặp địa điểm (không phân biệt thứ tự)
        
        Parameters:
        -----------
        location_id1: str
            ID của địa điểm 1
        location_id2: str
            ID của địa điểm 2
            
        Returns:
        --------
        tuple
            Khóa duy nhất cho cặp địa điểm
        """
        return tuple(sorted([str(location_id1), str(location_id2)]))
        
    def _clean_expired_events(self):
        """
        Xóa các sự kiện đã hết hạn
        """
        now = datetime.now()
        for user_id in list(self.recent_events.keys()):
            self.recent_events[user_id] = [
                event for event in self.recent_events[user_id]
                if (now - event['timestamp']).total_seconds() < self.event_expiry
            ]
            
    def get_realtime_recommendations(self, user_id, current_product_id=None, event_type='view', 
                                    product_details=None, collaborative_model=None, content_data=None, top_n=10):
        """
        Tạo gợi ý theo thời gian thực dựa trên sự kiện hiện tại và lịch sử gần đây
        
        Parameters:
        -----------
        user_id: int or str
            ID của người dùng
        current_product_id: int or str, optional
            ID địa điểm hiện tại người dùng đang xem
        event_type: str
            Loại sự kiện hiện tại
        product_details: DataFrame
            Chi tiết địa điểm
        collaborative_model: object
            Mô hình collaborative filtering
        content_data: tuple
            Dữ liệu content-based (similarity_matrix, index_to_id, id_to_index)
        top_n: int
            Số lượng gợi ý trả về
            
        Returns:
        --------
        list
            Danh sách các địa điểm được gợi ý với điểm số
        """
        # Khởi tạo danh sách gợi ý
        recommendations = []
        
        # Chuyển đổi ID thành string để so khớp với recent_events
        user_id = str(user_id)
        if current_product_id is not None:
            current_product_id = str(current_product_id)
        
        # 1. Gợi ý dựa trên địa điểm hiện tại (content-based)
        if current_product_id is not None and content_data:
            similarity_matrix, index_to_id, id_to_index = content_data
            
            try:
                if current_product_id in id_to_index:
                    product_idx = id_to_index[current_product_id]
                    # Lấy top N địa điểm tương tự
                    similar_indices = np.argsort(similarity_matrix[product_idx])[-(top_n+1):][::-1]
                    similar_indices = similar_indices[similar_indices != product_idx]  # Loại bỏ sản phẩm hiện tại
                    
                    for idx in similar_indices[:top_n]:
                        item_id = index_to_id[idx]
                        similarity_score = similarity_matrix[product_idx, idx]
                        recommendations.append({
                            'location_id': item_id,
                            'score': float(similarity_score),
                            'source': 'content'
                        })
            except Exception as e:
                logger.error(f"Error generating content-based recommendations: {str(e)}")
        
        # 2. Gợi ý dựa trên ma trận đồng xuất hiện (co-view, co-book)
        cooccurrence_recs = []
        
        if current_product_id is not None:
            # Gợi ý dựa trên co-view matrix
            coview_items = []
            for pair, count in self.coview_matrix.items():
                if current_product_id in pair:
                    other_item = pair[0] if pair[1] == current_product_id else pair[1]
                    coview_items.append((other_item, count))
            
            # Sắp xếp theo số lần xem cùng nhau
            coview_items.sort(key=lambda x: x[1], reverse=True)
            
            for item_id, count in coview_items[:top_n]:
                score = min(count / 5.0, 1.0)  # Chuẩn hóa điểm về 0-1
                cooccurrence_recs.append({
                    'location_id': item_id,
                    'score': score,
                    'source': 'coview'
                })
            
            # Gợi ý dựa trên co-book matrix
            cobook_items = []
            for pair, count in self.cobook_matrix.items():
                if current_product_id in pair:
                    other_item = pair[0] if pair[1] == current_product_id else pair[1]
                    cobook_items.append((other_item, count))
            
            # Sắp xếp theo số lần đặt cùng nhau
            cobook_items.sort(key=lambda x: x[1], reverse=True)
            
            for item_id, count in cobook_items[:5]:
                score = min(count / 3.0, 1.0) * 1.2  # Chuẩn hóa điểm và tăng trọng số do đã đặt
                cooccurrence_recs.append({
                    'location_id': item_id,
                    'score': score,
                    'source': 'cobook'
                })
        
        # Thêm vào danh sách gợi ý, giới hạn số lượng gợi ý
        recommendations.extend(cooccurrence_recs)
        
        # 3. Gợi ý dựa trên collaborative filtering
        if collaborative_model is not None:
            try:
                # Dự đoán điểm số cho các địa điểm
                if product_details is not None:
                    # Lấy ID của tất cả địa điểm
                    all_location_ids = product_details['location_id'].astype(str).tolist()
                    
                    # Lấy các địa điểm đã có trong recommendations
                    existing_ids = set(r['location_id'] for r in recommendations)
                    
                    # Chọn ngẫu nhiên 20 địa điểm để dự đoán
                    import random
                    predict_ids = random.sample([lid for lid in all_location_ids if lid not in existing_ids], min(20, len(all_location_ids)))
                    
                    for location_id in predict_ids:
                        try:
                            # Dự đoán điểm số
                            prediction = collaborative_model.predict(uid=user_id, iid=location_id)
                            
                            # Thêm vào danh sách gợi ý
                            if hasattr(prediction, 'est'):
                                score = prediction.est / 5.0  # Chuẩn hóa về 0-1
                                recommendations.append({
                                    'location_id': location_id,
                                    'score': score,
                                    'source': 'collaborative'
                                })
                        except Exception as e:
                            # Bỏ qua lỗi dự đoán
                            pass
            except Exception as e:
                logger.error(f"Error generating collaborative recommendations: {str(e)}")
                
        # Loại bỏ trùng lặp và sắp xếp theo điểm số
        unique_recommendations = {}
        for rec in recommendations:
            location_id = rec['location_id']
            if location_id not in unique_recommendations or rec['score'] > unique_recommendations[location_id]['score']:
                unique_recommendations[location_id] = rec
        
        # Chuyển về danh sách và sắp xếp
        recommendations = list(unique_recommendations.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Giới hạn số lượng gợi ý trả về
        recommendations = recommendations[:top_n]
        
        # Thêm thông tin chi tiết về địa điểm nếu có
        if product_details is not None:
            # Chuyển location_id trong recommendations về kiểu dữ liệu phù hợp với product_details
            for rec in recommendations:
                location_id = rec['location_id']
                matches = product_details[product_details['location_id'].astype(str) == str(location_id)]
                
                if not matches.empty:
                    # Thêm các thông tin từ product_details
                    for col in matches.columns:
                        if col != 'location_id' and col not in rec:
                            rec[col] = matches.iloc[0][col]
        
        return recommendations
