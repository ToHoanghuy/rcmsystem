"""
Script để kiểm tra tích hợp realtime
"""
import sys
import os

# Thêm đường dẫn hiện tại vào Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from recommenders.realtime import RealtimeRecommender
from utils.realtime_integration import RealtimeIntegration
from datetime import datetime
import pandas as pd

# Dữ liệu giả cho việc kiểm tra
product_data = {
    'product_id': [1, 2, 3, 4, 5],
    'name': ['Sản phẩm 1', 'Sản phẩm 2', 'Sản phẩm 3', 'Sản phẩm 4', 'Sản phẩm 5'],
    'rating': [4.5, 3.8, 4.2, 4.0, 3.9]
}
product_df = pd.DataFrame(product_data)

# Tạo mô hình giả
class DummySocketIO:
    def emit(self, event, data, room=None):
        print(f"Emitting {event} to {room}: {data}")

class DummyCollaborative:
    def predict(self, uid, iid):
        class Prediction:
            def __init__(self, est):
                self.est = est
        return Prediction(4.0)

# Tạo dữ liệu content-based giả
def create_dummy_content_data():
    import numpy as np
    similarity_matrix = np.array([
        [1.0, 0.8, 0.2, 0.5, 0.1],
        [0.8, 1.0, 0.3, 0.4, 0.2],
        [0.2, 0.3, 1.0, 0.9, 0.7],
        [0.5, 0.4, 0.9, 1.0, 0.6],
        [0.1, 0.2, 0.7, 0.6, 1.0]
    ])
    index_to_id = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    id_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    return (similarity_matrix, index_to_id, id_to_index)

# Kiểm tra các phương thức
def test_realtime_integration():
    print("Testing RealtimeIntegration...")
    
    # Tạo các đối tượng giả
    socketio = DummySocketIO()
    recommender = RealtimeRecommender()
    collaborative_model = DummyCollaborative()
    content_data = create_dummy_content_data()
    
    # Tạo RealtimeIntegration
    integration = RealtimeIntegration(
        socketio=socketio,
        realtime_recommender=recommender,
        product_details=product_df,
        collaborative_model=collaborative_model,
        content_data=content_data
    )
    
    # Kiểm tra đăng ký người dùng
    user_id = 1
    socket_id = "socket_123"
    success = integration.register_user(user_id, socket_id)
    print(f"Register user: {success}")
    
    # Kiểm tra ghi nhận sự kiện
    success, recommendations = integration.record_event(
        user_id=user_id,
        product_id=2,
        event_type="view",
        timestamp=datetime.now()
    )
    print(f"Record event: {success}")
    print(f"Recommendations: {recommendations}")
    
    # Kiểm tra lấy gợi ý
    recommendations = integration.get_recommendations(user_id=user_id, context_item=3)
    print(f"Get recommendations: {len(recommendations) if recommendations is not None else 'None'}")
    
    # Kiểm tra cập nhật cho tất cả người dùng
    results = integration.update_recommendations_for_all_users()
    print(f"Update for all users: {results}")

if __name__ == "__main__":
    test_realtime_integration()
