import pandas as pd
import numpy as np
import logging
from datetime import datetime
from utils.json_utils import convert_to_json_serializable

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
        # Lưu trữ mối tương quan giữa sản phẩm xem cùng nhau (co-view matrix)
        self.coview_matrix = {}
        # Lưu trữ mối tương quan giữa sản phẩm mua cùng nhau (co-purchase matrix)
        self.copurchase_matrix = {}
        # Maximum số lượng sự kiện lưu trữ cho mỗi người dùng
        self.max_events_per_user = 20
        # Thời gian hết hạn cho sự kiện (tính bằng giây)
        self.event_expiry = 3600  # 1 giờ
        
    def record_event(self, user_id, product_id, event_type, timestamp=None):
        """
        Ghi nhận sự kiện người dùng mới
        
        Parameters:
        -----------
        user_id: int
            ID của người dùng
        product_id: int
            ID của sản phẩm
        event_type: str
            Loại sự kiện ('view', 'add_to_cart', 'purchase', etc.)
        timestamp: datetime, optional
            Thời gian xảy ra sự kiện, mặc định là thời gian hiện tại
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Khởi tạo danh sách sự kiện cho người dùng nếu chưa có
        if user_id not in self.recent_events:
            self.recent_events[user_id] = []
        
        # Thêm sự kiện mới
        self.recent_events[user_id].append({
            'product_id': product_id,
            'event_type': event_type,
            'timestamp': timestamp
        })
        
        # Giới hạn số lượng sự kiện lưu trữ
        if len(self.recent_events[user_id]) > self.max_events_per_user:
            self.recent_events[user_id] = self.recent_events[user_id][-self.max_events_per_user:]
        
        # Cập nhật ma trận đồng xuất hiện
        self._update_cooccurrence_matrices(user_id, product_id, event_type)
        
        logger.info(f"Recorded event for user {user_id}: {event_type} on product {product_id}")
        
    def _update_cooccurrence_matrices(self, user_id, current_product_id, event_type):
        """
        Cập nhật ma trận đồng xuất hiện dựa trên sự kiện mới
        """
        if user_id not in self.recent_events or len(self.recent_events[user_id]) < 2:
            return
            
        # Lấy các sản phẩm gần đây người dùng đã tương tác
        recent_products = [event['product_id'] for event in self.recent_events[user_id][:-1]]
        
        # Cập nhật ma trận tương quan
        if event_type == 'view':
            for prev_product_id in recent_products:
                if prev_product_id != current_product_id:
                    # Cập nhật co-view matrix
                    key = self._get_product_pair_key(prev_product_id, current_product_id)
                    self.coview_matrix[key] = self.coview_matrix.get(key, 0) + 1
        
        elif event_type == 'purchase':
            # Lấy các sản phẩm xem gần đây trước khi mua
            for event in self.recent_events[user_id]:
                if event['event_type'] == 'view' and event['product_id'] != current_product_id:
                    # Cập nhật co-purchase matrix
                    key = self._get_product_pair_key(event['product_id'], current_product_id)
                    self.copurchase_matrix[key] = self.copurchase_matrix.get(key, 0) + 1

    def _get_product_pair_key(self, product_id1, product_id2):
        """
        Tạo khóa duy nhất cho cặp sản phẩm (không phân biệt thứ tự)
        """
        return tuple(sorted([product_id1, product_id2]))
        
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
        user_id: int
            ID của người dùng
        current_product_id: int, optional
            ID sản phẩm hiện tại người dùng đang xem
        event_type: str
            Loại sự kiện hiện tại
        product_details: DataFrame
            Chi tiết sản phẩm
        collaborative_model: object
            Mô hình collaborative filtering
        content_data: tuple
            Dữ liệu content-based (similarity_matrix, index_to_id, id_to_index)
        top_n: int
            Số lượng gợi ý trả về
            
        Returns:
        --------
        list
            Danh sách các sản phẩm được gợi ý với điểm số
        """
        # Khởi tạo danh sách gợi ý
        recommendations = []
        
        # 1. Gợi ý dựa trên sản phẩm hiện tại (content-based)
        if current_product_id is not None and content_data:
            similarity_matrix, index_to_id, id_to_index = content_data
            
            if current_product_id in id_to_index:
                product_idx = id_to_index[current_product_id]
                # Lấy top N sản phẩm tương tự
                similar_indices = similarity_matrix[product_idx].argsort()[-(top_n+1):][::-1]
                
                # Loại bỏ sản phẩm hiện tại
                similar_indices = [idx for idx in similar_indices if index_to_id[idx] != current_product_id]
                
                # Thêm vào danh sách gợi ý
                for idx in similar_indices[:5]:  # Lấy 5 sản phẩm tương tự nhất
                    prod_id = index_to_id[idx]
                    similarity_score = similarity_matrix[product_idx, idx]
                    recommendations.append({
                        'product_id': prod_id,
                        'score': similarity_score * 0.8,  # Giảm trọng số
                        'source': 'content_based'
                    })
        
        # 2. Gợi ý dựa trên hành vi gần đây (co-occurrence)
        if event_type == 'view' and current_product_id:
            # Tìm sản phẩm thường xem cùng
            coview_products = []
            for pair, count in self.coview_matrix.items():
                if current_product_id in pair:
                    other_product = pair[0] if pair[1] == current_product_id else pair[1]
                    coview_products.append((other_product, count))
            
            # Sắp xếp theo số lần xuất hiện cùng
            coview_products.sort(key=lambda x: x[1], reverse=True)
            
            # Thêm vào danh sách gợi ý
            for prod_id, count in coview_products[:3]:
                normalized_score = min(1.0, count / 10.0)  # Chuẩn hóa điểm (tối đa là 1.0)
                recommendations.append({
                    'product_id': prod_id,
                    'score': normalized_score * 0.9,  # Điểm cao hơn với co-view
                    'source': 'coview'
                })
        
        elif event_type == 'add_to_cart' and current_product_id:
            # Tìm sản phẩm thường mua cùng
            copurchase_products = []
            for pair, count in self.copurchase_matrix.items():
                if current_product_id in pair:
                    other_product = pair[0] if pair[1] == current_product_id else pair[1]
                    copurchase_products.append((other_product, count))
            
            # Sắp xếp theo số lần xuất hiện cùng
            copurchase_products.sort(key=lambda x: x[1], reverse=True)
            
            # Thêm vào danh sách gợi ý
            for prod_id, count in copurchase_products[:3]:
                normalized_score = min(1.0, count / 5.0)  # Chuẩn hóa điểm (tối đa là 1.0)
                recommendations.append({
                    'product_id': prod_id,
                    'score': normalized_score * 1.2,  # Điểm cao nhất với co-purchase
                    'source': 'copurchase'
                })
        
        # 3. Thêm gợi ý từ collaborative (nếu có)
        if user_id and collaborative_model:
            try:
                # Lấy tất cả sản phẩm
                if product_details is not None:
                    all_products = product_details['product_id'].unique()
                    
                    # Lấy top 5 sản phẩm với điểm cao nhất từ collaborative filtering
                    collab_scores = {}
                    for pid in all_products:
                        if pid != current_product_id:
                            try:
                                pred = collaborative_model.predict(uid=user_id, iid=pid)
                                collab_scores[pid] = pred.est
                            except:
                                pass
                    
                    # Sắp xếp và lấy top 3
                    top_collab_products = sorted(collab_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    for pid, score in top_collab_products:
                        recommendations.append({
                            'product_id': pid,
                            'score': score / 5.0 * 0.7,  # Chuẩn hóa điểm và giảm trọng số
                            'source': 'collaborative'
                        })
            except Exception as e:
                logger.error(f"Error getting collaborative recommendations: {str(e)}")
        
        # 4. Loại bỏ trùng lặp và sắp xếp
        seen_products = set()
        unique_recommendations = []
        
        for rec in recommendations:
            pid = rec['product_id']
            if pid not in seen_products and pid != current_product_id:
                seen_products.add(pid)
                unique_recommendations.append(rec)
        
        sorted_recommendations = sorted(unique_recommendations, key=lambda x: x['score'], reverse=True)
          # 5. Thêm thông tin chi tiết nếu có
        if product_details is not None:
            for rec in sorted_recommendations:
                prod_info = product_details[product_details['product_id'] == rec['product_id']]
                if not prod_info.empty:
                    for col in prod_info.columns:
                        if col != 'product_id':
                            rec[col] = prod_info.iloc[0][col]
        
        # 6. Convert any NumPy types to standard Python types for JSON serialization
        final_recommendations = convert_to_json_serializable(sorted_recommendations[:top_n])
        
        return final_recommendations
