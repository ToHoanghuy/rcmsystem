import numpy as np
import pandas as pd
from recommenders.content_based import get_similar_products

def adaptive_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, 
                                   user_events=None, top_n=10):
    """
    Phương pháp hybrid cải tiến với trọng số thích nghi
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based (similarity_matrix, index_to_id, id_to_index)
    user_id: int
        ID của người dùng cần gợi ý
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    user_events: DataFrame, optional
        DataFrame chứa dữ liệu sự kiện người dùng
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
        
    Returns:
    --------
    recommended_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    """
    # Giải nén dữ liệu content model
    similarity_matrix, index_to_id, id_to_index = content_model_data
    
    # Lấy tất cả sản phẩm
    all_products = product_details['product_id'].tolist()
    
    # Số lượng sản phẩm người dùng đã đánh giá (để điều chỉnh trọng số)
    user_rating_count = 0
    try:
        user_rating_count = len(collaborative_model.trainset.ur[collaborative_model.trainset.to_inner_uid(user_id)])
    except:
        pass  # Người dùng mới, không có đánh giá
    
    # Điều chỉnh trọng số dựa trên số lượng đánh giá
    if user_rating_count > 10:
        collab_weight = 0.7  # Người dùng có nhiều đánh giá -> tin tưởng CF
    elif user_rating_count > 0:
        collab_weight = 0.5  # Người dùng có ít đánh giá -> cân bằng
    else:
        collab_weight = 0.3  # Người dùng mới -> tin tưởng CB hơn
    
    content_weight = 1 - collab_weight
    
    # Điều chỉnh thêm dựa trên sự kiện người dùng nếu có
    if user_events is not None and user_id in user_events.index:
        event_data = user_events.loc[user_id]
        # Người dùng có nhiều sự kiện -> tăng trọng số content-based
        if event_data.sum() > 5:
            collab_weight -= 0.1
            content_weight += 0.1
    
    # Đảm bảo trọng số trong khoảng [0, 1]
    collab_weight = max(0, min(1, collab_weight))
    content_weight = max(0, min(1, content_weight))
    
    print(f"Adaptive weights: Collaborative = {collab_weight:.2f}, Content = {content_weight:.2f}")
    
    # Tính điểm hybrid
    product_scores = {}
    for product_id in all_products:
        # Điểm collaborative filtering
        cf_score = 3.0  # Giá trị mặc định
        try:
            cf_score = collaborative_model.predict(uid=user_id, iid=product_id).est
        except:
            pass
        
        # Điểm content-based
        cb_score = 3.0  # Giá trị mặc định
        if product_id in id_to_index:
            product_idx = id_to_index[product_id]
            
            # Tìm các sản phẩm tương tự mà người dùng đã đánh giá
            similar_scores = []
            for other_id in all_products:
                if other_id != product_id and other_id in id_to_index:
                    other_idx = id_to_index[other_id]
                    try:
                        # Kiểm tra xem người dùng đã đánh giá sản phẩm này chưa
                        inner_uid = collaborative_model.trainset.to_inner_uid(user_id)
                        inner_iid = collaborative_model.trainset.to_inner_iid(other_id)
                        rating = collaborative_model.trainset.ur[inner_uid]
                        
                        for item_idx, rating_val in rating:
                            if item_idx == inner_iid:
                                # Nếu đã đánh giá, tính điểm dựa trên đánh giá và độ tương đồng
                                similarity = similarity_matrix[product_idx, other_idx]
                                similar_scores.append((similarity, rating_val))
                                break
                    except:
                        continue
            
            # Tính điểm content-based nếu có dữ liệu
            if similar_scores:
                cb_score = sum(sim * rating for sim, rating in similar_scores) / sum(sim for sim, _ in similar_scores)
            else:
                # Nếu không có dữ liệu, sử dụng điểm tương đồng trung bình với các sản phẩm khác
                cb_score = np.mean([similarity_matrix[product_idx, other_idx] for other_idx in range(len(similarity_matrix)) if other_idx != product_idx]) * 5
        
        # Tính điểm hybrid
        hybrid_score = collab_weight * cf_score + content_weight * cb_score
        product_scores[product_id] = hybrid_score
    
    # Lấy top N sản phẩm với điểm cao nhất
    top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_product_ids = [product_id for product_id, _ in top_products]
    
    # Lấy thông tin chi tiết và thêm điểm
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    score_dict = {pid: score for pid, score in top_products}
    recommended_products['hybrid_score'] = recommended_products['product_id'].map(score_dict)
    recommended_products = recommended_products.sort_values(by='hybrid_score', ascending=False)
    
    return recommended_products

def switching_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, confidence_threshold=0.8, top_n=10):
    """
    Phương pháp switching hybrid: chọn CF hoặc CB tùy theo độ tin cậy
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based (similarity_matrix, index_to_id, id_to_index)
    user_id: int
        ID của người dùng cần gợi ý
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    confidence_threshold: float
        Ngưỡng độ tin cậy để quyết định sử dụng CF hay CB
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
        
    Returns:
    --------
    recommended_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    recommendation_type: str
        Loại gợi ý đã được chọn ("collaborative" hoặc "content_based")
    """
    # Giải nén dữ liệu content model
    similarity_matrix, index_to_id, id_to_index = content_model_data
    
    # Kiểm tra độ tin cậy của CF cho người dùng này
    cf_confidence = 0.0
    try:
        user_inner_id = collaborative_model.trainset.to_inner_uid(user_id)
        num_ratings = len(collaborative_model.trainset.ur[user_inner_id])
        cf_confidence = min(1.0, num_ratings / 10.0)  # Độ tin cậy tỷ lệ với số lượng đánh giá
    except:
        cf_confidence = 0.0  # Người dùng mới, không có đánh giá
    
    print(f"Collaborative filtering confidence: {cf_confidence:.2f}")
    
    # Quyết định sử dụng CF hay CB
    if cf_confidence > confidence_threshold:
        print(f"Using collaborative filtering (confidence > {confidence_threshold})")
        # Sử dụng Collaborative Filtering
        product_scores = {}
        for product_id in product_details['product_id'].tolist():
            try:
                score = collaborative_model.predict(uid=user_id, iid=product_id).est
                product_scores[product_id] = score
            except:
                continue  # Bỏ qua nếu không dự đoán được
        
        # Lấy top N sản phẩm
        top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_product_ids = [product_id for product_id, _ in top_products]
        
        # Lấy thông tin chi tiết
        recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
        score_dict = {pid: score for pid, score in top_products}
        recommended_products['score'] = recommended_products['product_id'].map(score_dict)
        recommended_products = recommended_products.sort_values(by='score', ascending=False)
        
        return recommended_products, "collaborative"
    else:
        print(f"Using content-based filtering (confidence <= {confidence_threshold})")
        # Sử dụng Content-Based
        # Tìm các sản phẩm mà người dùng đã đánh giá cao
        user_liked_products = []
        try:
            user_inner_id = collaborative_model.trainset.to_inner_uid(user_id)
            for item_idx, rating in collaborative_model.trainset.ur[user_inner_id]:
                if rating > 3.5:  # Ngưỡng cho sản phẩm được thích
                    product_id = collaborative_model.trainset.to_raw_iid(item_idx)
                    user_liked_products.append(product_id)
        except:
            # Nếu không có đánh giá, sử dụng sản phẩm phổ biến
            top_rated_products = product_details.sort_values(by='rating', ascending=False).head(3)
            user_liked_products = top_rated_products['product_id'].tolist()
        
        # Tìm các sản phẩm tương tự
        similar_products = pd.DataFrame()
        for liked_product in user_liked_products:
            if (liked_product in id_to_index):
                similar = get_similar_products(
                    liked_product, 
                    similarity_matrix, 
                    id_to_index, 
                    index_to_id, 
                    product_details, 
                    top_n=5
                )
                similar_products = pd.concat([similar_products, similar])
        
        # Loại bỏ trùng lặp và lấy top N
        similar_products = similar_products.drop_duplicates(subset=['product_id'])
        similar_products = similar_products.sort_values(by='similarity_score', ascending=False).head(top_n)
        
        return similar_products, "content_based"

def hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, method="adaptive", top_n=10, user_events=None):
    """
    Gợi ý kết hợp sử dụng cả Collaborative Filtering và Content-Based Filtering
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based (similarity_matrix, index_to_id, id_to_index)
    user_id: int
        ID của người dùng cần gợi ý
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    method: str
        Phương pháp hybrid: "adaptive" (trọng số thích nghi) hoặc "switching" (chuyển đổi)
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
    user_events: DataFrame, optional
        DataFrame chứa dữ liệu sự kiện người dùng
        
    Returns:
    --------
    recommended_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    """
    if method == "adaptive":
        return adaptive_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, user_events, top_n)
    elif method == "switching":
        recommended_products, _ = switching_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, top_n=top_n)
        return recommended_products
    else:
        raise ValueError("Method must be either 'adaptive' or 'switching'")

class MockCollaborativeModel:
    def predict(self, uid, iid):
        return type('MockPrediction', (object,), {'est': 4.5})()

collaborative_model = MockCollaborativeModel()