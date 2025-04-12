import numpy as np

def hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details, num_recommendations=10):
    # Lấy danh sách tất cả các sản phẩm
    product_ids = product_details['product_id'].tolist()
    
    # Dự đoán rating từ Collaborative Filtering
    collaborative_scores = []
    for product_id in product_ids:
        prediction = collaborative_model.predict(uid=user_id, iid=product_id)
        collaborative_scores.append(prediction.est)
    
    # Chuyển đổi thành mảng numpy
    collaborative_scores = np.array(collaborative_scores)
    
    # Lấy điểm tương đồng từ Content-Based Filtering
    if isinstance(content_similarity, np.ndarray) and user_id < content_similarity.shape[0]:
        content_scores = content_similarity[user_id]
    else:
        content_scores = np.zeros(len(product_ids))  # Nếu không có điểm tương đồng, gán 0
    
    # Kết hợp điểm số từ hai mô hình
    combined_scores = collaborative_scores + content_scores
    
    # Lấy top N sản phẩm có điểm số cao nhất
    top_indices = combined_scores.argsort()[-num_recommendations:][::-1]
    return product_details.iloc[top_indices]  

class MockCollaborativeModel:
    def predict(self, uid, iid):
        return type('MockPrediction', (object,), {'est': 4.5})()

collaborative_model = MockCollaborativeModel()