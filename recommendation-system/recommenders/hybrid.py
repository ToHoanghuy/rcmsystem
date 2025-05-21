import numpy as np
import pandas as pd
from recommenders.content_based import get_similar_products
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Điều chỉnh trọng số dựa trên số lượng đánh giá - Cải tiến với công thức sigmoid
    # Giúp làm mượt hơn việc chuyển đổi trọng số khi số lượng đánh giá của người dùng tăng
    collab_weight = 1 / (1 + np.exp(-0.3 * (user_rating_count - 5)))
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
            try:
                cf_score = collaborative_model.predict(uid=user_id, iid=product_id).est
            except (ValueError, TypeError) as e:
                # Nếu lỗi kiểu dữ liệu, thử với dạng chuỗi
                print(f"Converting IDs to string: {e}")
                cf_score = collaborative_model.predict(uid=str(user_id), iid=str(product_id)).est
        except Exception as e:
            print(f"Hybrid prediction failed for user {user_id}, product {product_id}: {e}")
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

def context_aware_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, 
                                        user_events=None, user_context=None, top_n=10):
    """
    Phương pháp hybrid mới có nhận thức ngữ cảnh (context-aware)
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based
    user_id: int
        ID của người dùng cần gợi ý
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    user_events: DataFrame, optional
        DataFrame chứa dữ liệu sự kiện người dùng
    user_context: dict, optional
        Dictionary chứa thông tin ngữ cảnh của người dùng (thời gian, vị trí, thiết bị, v.v.)
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
    
    # Khởi tạo trọng số cơ bản
    collab_weight = 0.5
    content_weight = 0.5
    
    # --- CẢI TIẾN 1: Điều chỉnh trọng số dựa trên ngữ cảnh ---
    if user_context is not None:
        # Điều chỉnh theo thời gian trong ngày nếu có
        if 'time_of_day' in user_context:
            time = user_context['time_of_day']
            # Buổi sáng và tối: tăng content-based (giả định người dùng muốn khám phá nội dung mới)
            if time in ['morning', 'evening']:
                content_weight += 0.1
                collab_weight -= 0.1
            # Trưa và chiều: tăng collaborative (giả định người dùng có xu hướng theo đám đông)
            elif time in ['afternoon', 'noon']:
                collab_weight += 0.1
                content_weight -= 0.1
                
        # Điều chỉnh theo thiết bị nếu có
        if 'device' in user_context:
            device = user_context['device']
            # Mobile: tăng collaborative (giả định người dùng mobile thích các đề xuất phổ biến)
            if device == 'mobile':
                collab_weight += 0.1
                content_weight -= 0.1
            # Desktop: tăng content-based (giả định người dùng desktop thích khám phá chi tiết)
            elif device == 'desktop':
                content_weight += 0.1
                collab_weight -= 0.1
    
    # --- CẢI TIẾN 2: Phân tích lịch sử tương tác để cá nhân hóa trọng số ---
    if user_events is not None and user_id in user_events.index:
        event_data = user_events.loc[user_id]
        # Nếu người dùng có nhiều sự kiện tìm kiếm/khám phá -> tăng trọng số content-based
        if 'search_events' in event_data and event_data['search_events'] > 3:
            content_weight += 0.1
            collab_weight -= 0.1
        
        # Nếu người dùng thường xuyên mua hàng phổ biến -> tăng trọng số collaborative
        if 'purchase_popular_items' in event_data and event_data['purchase_popular_items'] > 3:
            collab_weight += 0.1
            content_weight -= 0.1
    
    # Đảm bảo trọng số trong khoảng [0, 1]
    collab_weight = max(0, min(1, collab_weight))
    content_weight = max(0, min(1, content_weight))
    
    print(f"Context-aware weights: Collaborative = {collab_weight:.2f}, Content = {content_weight:.2f}")
    
    # Tính điểm hybrid tương tự như adaptive_hybrid_recommendations...
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
            try:
                inner_uid = collaborative_model.trainset.to_inner_uid(user_id)
                for item_idx, rating_val in collaborative_model.trainset.ur[inner_uid]:
                    other_id = collaborative_model.trainset.to_raw_iid(item_idx)
                    if other_id != product_id and other_id in id_to_index:
                        other_idx = id_to_index[other_id]
                        similarity = similarity_matrix[product_idx, other_idx]
                        similar_scores.append((similarity, rating_val))
            except:
                pass
            
            # Tính điểm content-based nếu có dữ liệu
            if similar_scores:
                cb_score = sum(sim * rating for sim, rating in similar_scores) / sum(sim for sim, _ in similar_scores)
            else:
                # Nếu không có dữ liệu, sử dụng điểm tương đồng trung bình với các sản phẩm khác
                cb_score = np.mean([similarity_matrix[product_idx, other_idx] for other_idx in range(len(similarity_matrix)) if other_idx != product_idx]) * 5
        
        # --- CẢI TIẾN 3: Tích hợp phong cách người dùng vào điểm số ---
        user_style_bonus = 0
        if user_context and 'preferred_categories' in user_context:
            # Kiểm tra sản phẩm có thuộc danh mục ưa thích của người dùng
            if 'category' in product_details.columns:
                product_category = product_details.loc[product_details['product_id'] == product_id, 'category'].values
                if len(product_category) > 0 and product_category[0] in user_context['preferred_categories']:
                    user_style_bonus = 0.5  # Tăng điểm cho sản phẩm thuộc danh mục ưa thích
        
        # Tính điểm hybrid
        hybrid_score = collab_weight * cf_score + content_weight * cb_score + user_style_bonus
        product_scores[product_id] = hybrid_score
    
    # --- CẢI TIẾN 4: Tăng độ đa dạng trong kết quả đề xuất ---
    # Lấy top sản phẩm với thuật toán MMR (Maximal Marginal Relevance)
    # để cân bằng giữa độ liên quan và độ đa dạng
    lambda_param = 0.7  # Tham số cân bằng giữa độ liên quan (relevance) và độ đa dạng (diversity)
    
    selected_items = []
    remaining_items = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    while len(selected_items) < top_n and remaining_items:
        if not selected_items:
            # Chọn item đầu tiên là item có điểm số cao nhất
            selected_items.append(remaining_items[0])
            remaining_items = remaining_items[1:]
        else:
            # Tính MMR cho các item còn lại
            mmr_scores = []
            
            for i, (item_id, score) in enumerate(remaining_items):
                if item_id not in id_to_index:
                    mmr_scores.append((i, -float('inf')))
                    continue
                
                item_idx = id_to_index[item_id]
                
                # Độ liên quan (relevance)
                relevance = score
                
                # Độ đa dạng (diversity) - độ khác biệt với các item đã chọn
                max_similarity = 0
                for selected_item_id, _ in selected_items:
                    if selected_item_id in id_to_index:
                        selected_idx = id_to_index[selected_item_id]
                        sim = similarity_matrix[item_idx, selected_idx]
                        max_similarity = max(max_similarity, sim)
                
                # Tính điểm MMR: lambda * relevance - (1 - lambda) * max_similarity
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append((i, mmr))
            
            # Chọn item có điểm MMR cao nhất
            best_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_items.append(remaining_items[best_idx])
            remaining_items = remaining_items[:best_idx] + remaining_items[best_idx+1:]
    
    # Trả về kết quả
    top_product_ids = [product_id for product_id, _ in selected_items]
    scores = {product_id: score for product_id, score in selected_items}
    
    # Lấy thông tin chi tiết và thêm điểm
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    recommended_products['hybrid_score'] = recommended_products['product_id'].map(scores)
    recommended_products = recommended_products.sort_values(by='hybrid_score', ascending=False)
    
    return recommended_products

def personalized_matrix_factorization_hybrid(collaborative_model, content_model_data, user_id, product_details, 
                                           user_events=None, top_n=10):
    """
    Phương pháp hybrid cải tiến sử dụng phân rã ma trận cá nhân hóa
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based
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
    
    # Xây dựng ma trận đánh giá cho người dùng
    user_ratings = {}
    try:
        inner_uid = collaborative_model.trainset.to_inner_uid(user_id)
        for item_idx, rating in collaborative_model.trainset.ur[inner_uid]:
            item_id = collaborative_model.trainset.to_raw_iid(item_idx)
            user_ratings[item_id] = rating
    except:
        pass
    
    # --- CẢI TIẾN 1: Sử dụng SVD để phân rã ma trận tương đồng content-based ---
    # Tạo ma trận đặc trưng cho các sản phẩm mà người dùng đã đánh giá
    rated_products_indices = [id_to_index[pid] for pid in user_ratings.keys() if pid in id_to_index]
    
    if rated_products_indices:
        # Trích xuất các vector đặc trưng cho sản phẩm đã đánh giá
        rated_products_features = similarity_matrix[rated_products_indices]
        
        # Phân tích SVD để tìm các yếu tố ẩn
        if len(rated_products_indices) > 1:  # Cần ít nhất 2 sản phẩm để phân tích có ý nghĩa
            try:
                # Sử dụng SVD để phân rã ma trận đặc trưng
                num_factors = min(3, len(rated_products_indices) - 1)  # Số lượng yếu tố ẩn
                U, sigma, Vt = svds(rated_products_features, k=num_factors)
                
                # Xây dựng ma trận trọng số cho từng sản phẩm dựa trên các yếu tố ẩn
                feature_weights = np.dot(np.diag(sigma), Vt)
                
                # Chuẩn hóa trọng số
                feature_weights = feature_weights / np.linalg.norm(feature_weights, axis=0, keepdims=True)
                
                # Tính toán trọng số cho content và collaborative dựa trên phân tích
                # Độ tương đồng giữa các sản phẩm đã đánh giá và vector đặc trưng
                ratings = np.array([user_ratings[collaborative_model.trainset.to_raw_iid(collaborative_model.trainset.to_inner_iid(pid))] 
                               for pid in user_ratings.keys() if pid in id_to_index])
                
                # Ma trận phương sai để xác định độ tin cậy của mô hình content-based
                content_confidence = min(1.0, np.mean(np.var(rated_products_features, axis=0)) * 2)
                
                # Điều chỉnh trọng số dựa trên độ tin cậy
                content_weight = 0.4 + content_confidence * 0.2  # Trong khoảng [0.4, 0.6]
                collab_weight = 1 - content_weight
            except:
                collab_weight = 0.5
                content_weight = 0.5
        else:
            collab_weight = 0.5
            content_weight = 0.5
    else:
        # Không có dữ liệu đánh giá, thiên về content-based
        collab_weight = 0.3
        content_weight = 0.7
    
    print(f"Matrix factorization weights: Collaborative = {collab_weight:.2f}, Content = {content_weight:.2f}")
    
    # --- CẢI TIẾN 2: Kết hợp cả hai mô hình với trọng số được tối ưu hóa ---
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
            try:
                inner_uid = collaborative_model.trainset.to_inner_uid(user_id)
                for item_idx, rating_val in collaborative_model.trainset.ur[inner_uid]:
                    other_id = collaborative_model.trainset.to_raw_iid(item_idx)
                    if other_id != product_id and other_id in id_to_index:
                        other_idx = id_to_index[other_id]
                        similarity = similarity_matrix[product_idx, other_idx]
                        similar_scores.append((similarity, rating_val))
            except:
                pass
            
            # Tính điểm content-based nếu có dữ liệu
            if similar_scores:
                # Cải tiến: Sử dụng weighted average với trọng số là độ tương đồng bình phương
                # Giúp tăng tầm ảnh hưởng của các sản phẩm rất tương đồng, giảm nhiễu
                cb_score = sum((sim**2) * rating for sim, rating in similar_scores) / sum((sim**2) for sim, _ in similar_scores)
            else:
                # Nếu không có dữ liệu, sử dụng điểm tương đồng trung bình với các sản phẩm khác
                cb_score = np.mean([similarity_matrix[product_idx, other_idx] for other_idx in range(len(similarity_matrix)) if other_idx != product_idx]) * 5
        
        # --- CẢI TIẾN 3: Thêm yếu tố mới mẻ (novelty) và phổ biến (popularity) ---
        popularity_score = 0
        if 'popularity' in product_details.columns:
            # Chuẩn hóa điểm phổ biến trong khoảng [0, 1]
            max_pop = product_details['popularity'].max()
            if max_pop > 0:
                popularity = product_details.loc[product_details['product_id'] == product_id, 'popularity'].values
                if len(popularity) > 0:
                    popularity_score = popularity[0] / max_pop
        
        # Trọng số cho yếu tố phổ biến (0.1), đảm bảo không ảnh hưởng quá nhiều
        novelty_factor = 0.1 * (1 - popularity_score)  # Sản phẩm càng ít phổ biến càng mới mẻ
        
        # Tính điểm hybrid có tính đến yếu tố mới mẻ
        hybrid_score = (collab_weight * cf_score + content_weight * cb_score) * (1 + novelty_factor)
        product_scores[product_id] = hybrid_score
    
    # --- CẢI TIẾN 4: Lựa chọn sản phẩm đề xuất với thuật toán greedy re-ranking ---
    # Đây là phương pháp tìm tập con tối ưu của top-N sản phẩm để tối đa hóa cả điểm số và độ đa dạng
    alpha = 0.3  # Trọng số cho độ đa dạng (0.3 = 30% cho độ đa dạng, 70% cho điểm số)
    
    # Sắp xếp sản phẩm theo điểm số
    sorted_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Bắt đầu với sản phẩm có điểm cao nhất
    selected_products = [sorted_products[0]]
    remaining_products = sorted_products[1:]
    
    while len(selected_products) < top_n and remaining_products:
        max_score = -float('inf')
        best_product_idx = 0
        
        # Tìm sản phẩm tốt nhất để thêm vào danh sách đã chọn
        for i, (product_id, score) in enumerate(remaining_products):
            if product_id not in id_to_index:
                continue
                
            product_idx = id_to_index[product_id]
            
            # Tính độ đa dạng của sản phẩm này so với các sản phẩm đã chọn
            avg_similarity = 0
            for selected_product_id, _ in selected_products:
                if selected_product_id in id_to_index:
                    selected_idx = id_to_index[selected_product_id]
                    avg_similarity += similarity_matrix[product_idx, selected_idx]
            
            if len(selected_products) > 0:
                avg_similarity /= len(selected_products)
            
            # Tính điểm kết hợp: (1-alpha) * điểm số + alpha * (1 - độ tương đồng trung bình)
            combined_score = (1 - alpha) * score + alpha * (1 - avg_similarity)
            
            if combined_score > max_score:
                max_score = combined_score
                best_product_idx = i
        
        # Thêm sản phẩm tốt nhất vào danh sách đã chọn
        selected_products.append(remaining_products[best_product_idx])
        remaining_products.pop(best_product_idx)
    
    # Trả về kết quả
    top_product_ids = [product_id for product_id, _ in selected_products]
    scores = {product_id: score for product_id, score in selected_products}
    
    # Lấy thông tin chi tiết và thêm điểm
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    recommended_products['hybrid_score'] = recommended_products['product_id'].map(scores)
    recommended_products = recommended_products.sort_values(by='hybrid_score', ascending=False)
    
    # Tính độ đa dạng của kết quả cuối cùng
    diversity = calculate_diversity(recommended_products['product_id'].tolist(), id_to_index, similarity_matrix)
    print(f"Result diversity: {diversity:.3f}")
    
    return recommended_products

def calculate_diversity(product_ids, id_to_index, similarity_matrix):
    """
    Tính độ đa dạng của một tập các sản phẩm dựa trên ma trận độ tương đồng
    
    Độ đa dạng được tính dựa trên độ khác biệt trung bình giữa các cặp sản phẩm
    Giá trị càng cao nghĩa là tập sản phẩm càng đa dạng (càng khác biệt nhau)
    
    Parameters:
    -----------
    product_ids: list
        Danh sách ID của các sản phẩm cần tính độ đa dạng
    id_to_index: dict
        Dictionary ánh xạ từ product_id sang index trong ma trận tương đồng
    similarity_matrix: numpy.ndarray
        Ma trận độ tương đồng giữa các sản phẩm
        
    Returns:
    --------
    diversity_score: float
        Điểm đa dạng của tập sản phẩm, trong khoảng [0, 1]
        0 = tất cả sản phẩm giống nhau, 1 = tất cả sản phẩm hoàn toàn khác nhau
    """
    valid_ids = [pid for pid in product_ids if pid in id_to_index]
    
    if len(valid_ids) < 2:
        return 0.0
    
    # Lấy các chỉ số tương ứng trong ma trận tương đồng
    indices = [id_to_index[pid] for pid in valid_ids]
    
    # Tính tổng độ tương đồng giữa các cặp sản phẩm
    total_similarity = 0.0
    count = 0
    
    for i, idx1 in enumerate(indices):
        for j, idx2 in enumerate(indices):
            if i < j:  # Chỉ tính một lần cho mỗi cặp
                total_similarity += similarity_matrix[idx1, idx2]
                count += 1
    
    if count > 0:
        # Tính độ tương đồng trung bình
        avg_similarity = total_similarity / count
        # Chuyển từ độ tương đồng sang độ đa dạng (đảo ngược)
        diversity_score = 1.0 - avg_similarity
        return diversity_score
    else:
        return 0.0

def hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, method="context_aware", top_n=10, user_events=None, user_context=None):
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
        Phương pháp hybrid: "adaptive", "switching", "context_aware", "matrix_factorization", "precision_boosted"
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
    user_events: DataFrame, optional
        DataFrame chứa dữ liệu sự kiện người dùng
    user_context: dict, optional
        Dictionary chứa thông tin ngữ cảnh của người dùng
        
    Returns:
    --------
    recommended_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    recommendation_type: str
        Loại gợi ý đã được chọn (chỉ trả về khi method="switching")
    """
    # Thêm phương thức mới tập trung vào việc tăng precision
    if method == "precision_boosted":
        return precision_boosted_hybrid_recommendations(collaborative_model, content_model_data, user_id, 
                                                      product_details, user_events, top_n)
    elif method == "adaptive":
        return adaptive_hybrid_recommendations(collaborative_model, content_model_data, user_id, 
                                             product_details, user_events, top_n)
    elif method == "switching":
        return switching_hybrid_recommendations(collaborative_model, content_model_data, user_id, 
                                              product_details, 0.8, top_n)
    elif method == "context_aware":
        return context_aware_hybrid_recommendations(collaborative_model, content_model_data, user_id,
                                                  product_details, user_events, user_context, top_n)
    elif method == "matrix_factorization":
        return personalized_matrix_factorization_hybrid(collaborative_model, content_model_data, user_id,
                                                      product_details, user_events, top_n)
    else:
        raise ValueError(f"Phương pháp hybrid không hợp lệ: {method}")

def precision_boosted_hybrid_recommendations(collaborative_model, content_model_data, user_id, product_details, 
                                          user_events=None, top_n=10):
    """
    Phương pháp hybrid cải tiến tập trung vào tăng precision bằng cách sử dụng ngưỡng tin cậy cao hơn
    và kết hợp nhiều nguồn thông tin với trọng số tối ưu
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data: tuple
        Chứa dữ liệu của mô hình Content-Based
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
    
    # --- CÁCH TIẾP CẬN 1: Kết hợp dữ liệu người dùng và đặc tính sản phẩm ---
    
    # Phân tích lịch sử người dùng để hiểu sở thích
    user_ratings = {}
    user_avg_rating = 3.0  # Giá trị mặc định
    user_rating_count = 0
    user_high_rated_products = []
    product_types_liked = set()
    price_range_liked = []
    
    try:
        # Lấy các đánh giá của người dùng
        inner_uid = collaborative_model.trainset.to_inner_uid(user_id)
        for item_idx, rating_val in collaborative_model.trainset.ur[inner_uid]:
            product_id = collaborative_model.trainset.to_raw_iid(item_idx)
            user_ratings[product_id] = rating_val
            
            # Lưu sản phẩm có đánh giá cao
            if rating_val >= 4.0:
                user_high_rated_products.append(product_id)
                
                # Lấy thông tin về loại sản phẩm và giá được ưa thích
                product_row = product_details[product_details['product_id'] == product_id]
                if not product_row.empty:
                    if 'product_type' in product_row.columns:
                        product_types_liked.add(product_row['product_type'].iloc[0])
                    if 'price' in product_row.columns:
                        price = product_row['price'].iloc[0]
                        if not pd.isna(price) and price > 0:
                            price_range_liked.append(price)
        
        user_rating_count = len(user_ratings)
        if user_rating_count > 0:
            user_avg_rating = sum(user_ratings.values()) / user_rating_count
    except:
        # Người dùng mới không có đánh giá
        pass
    
    # Xác định phạm vi giá ưa thích nếu có
    min_price, max_price = 0, float('inf')
    if len(price_range_liked) >= 3:  # Cần ít nhất 3 điểm dữ liệu để xác định phạm vi giá một cách đáng tin cậy
        min_price = max(0, np.percentile(price_range_liked, 25) * 0.8)  # Mở rộng xuống 20%
        max_price = np.percentile(price_range_liked, 75) * 1.2  # Mở rộng lên 20%
    
    # --- CÁCH TIẾP CẬN 2: Nâng cao độ chính xác bằng cách đặt ngưỡng tin cậy cho đề xuất ---
    
    # Tính điểm cho từng sản phẩm với nhiều yếu tố
    product_scores = {}
    product_confidence = {}
    
    for product_id in all_products:
        # Bỏ qua các sản phẩm người dùng đã đánh giá
        if product_id in user_ratings:
            continue
            
        # Đánh giá theo CF
        cf_score = None
        cf_confidence = 0.0
        
        try:
            # Thử lấy dự đoán từ mô hình
            prediction = collaborative_model.predict(uid=user_id, iid=product_id)
            cf_score = prediction.est
            
            # Tính độ tin cậy dựa trên số lượng đánh giá của người dùng và độ tương đồng với người dùng khác
            if hasattr(prediction, 'details') and 'was_impossible' in prediction.details:
                if prediction.details['was_impossible']:
                    cf_confidence = 0.0
                else:
                    # Người dùng có nhiều đánh giá thì độ tin cậy cao hơn
                    cf_confidence = min(0.9, 0.3 + 0.1 * min(6, user_rating_count))
            else:
                cf_confidence = min(0.9, 0.3 + 0.1 * min(6, user_rating_count))
        except:
            # Nếu không lấy được dự đoán từ CF
            cf_score = None
        
        # Đánh giá theo thông tin sản phẩm và sở thích người dùng
        product_confidence = 0.0
        similarity_score = 0.0
        
        # Lấy thông tin sản phẩm
        product_row = product_details[product_details['product_id'] == product_id]
        if not product_row.empty:
            # Kiểm tra loại sản phẩm
            if 'product_type' in product_row.columns and product_types_liked:
                product_type = product_row['product_type'].iloc[0]
                if product_type in product_types_liked:
                    # Tăng độ tin cậy và điểm số cho sản phẩm cùng loại
                    product_confidence += 0.2
                    similarity_score += 0.5
                    
            # Kiểm tra phạm vi giá
            if 'price' in product_row.columns and len(price_range_liked) >= 3:
                price = product_row['price'].iloc[0]
                if not pd.isna(price) and min_price <= price <= max_price:
                    # Tăng độ tin cậy và điểm số cho sản phẩm trong phạm vi giá ưa thích
                    product_confidence += 0.15
                    similarity_score += 0.3
            
            # Kiểm tra rating trung bình sản phẩm
            if 'rating' in product_row.columns:
                avg_rating = product_row['rating'].iloc[0]
                if not pd.isna(avg_rating) and avg_rating > 4.0:
                    # Tăng độ tin cậy và điểm số cho sản phẩm có rating cao
                    product_confidence += 0.15
                    similarity_score += avg_rating / 5.0 * 0.2
        
        # Đánh giá tương đồng với các sản phẩm được đánh giá cao
        content_score = 0.0
        content_confidence = 0.0
        
        if product_id in id_to_index and user_high_rated_products:
            product_idx = id_to_index[product_id]
            
            # Tính điểm tương đồng trung bình có trọng số với các sản phẩm yêu thích
            weighted_similarities = []
            
            for liked_product_id in user_high_rated_products:
                if liked_product_id in id_to_index:
                    liked_idx = id_to_index[liked_product_id]
                    similarity = similarity_matrix[product_idx, liked_idx]
                    liked_rating = user_ratings[liked_product_id]
                    weighted_similarities.append((similarity, liked_rating))
            
            if weighted_similarities:
                # Tính điểm content có trọng số
                total_weight = sum(sim for sim, _ in weighted_similarities)
                if total_weight > 0:
                    content_score = sum(sim * rating for sim, rating in weighted_similarities) / total_weight
                    # Độ tin cậy phụ thuộc vào mức độ tương đồng và số lượng sản phẩm tương tự
                    content_confidence = min(0.8, 0.2 + 0.1 * len(weighted_similarities) + 0.5 * total_weight / len(weighted_similarities))
        
        # Kết hợp các điểm số với trọng số thích hợp
        final_score = None
        confidence = 0.0
        
        if cf_score is not None:
            if content_score > 0:
                # Kết hợp cả CF và content-based với trọng số dựa trên độ tin cậy
                hybrid_weight = cf_confidence / (cf_confidence + content_confidence + product_confidence + 0.01)
                content_weight = content_confidence / (cf_confidence + content_confidence + product_confidence + 0.01)
                product_weight = product_confidence / (cf_confidence + content_confidence + product_confidence + 0.01)
                
                final_score = (hybrid_weight * cf_score + 
                              content_weight * content_score + 
                              product_weight * similarity_score)
                
                confidence = cf_confidence + content_confidence + product_weight
            else:
                # Chỉ có CF
                final_score = cf_score
                confidence = cf_confidence
        elif content_score > 0:
            # Chỉ có content-based
            final_score = content_score
            confidence = content_confidence
        elif similarity_score > 0:
            # Chỉ có thông tin sản phẩm
            # Sử dụng user_avg_rating để điều chỉnh dự đoán dựa trên thông tin sản phẩm
            final_score = user_avg_rating * 0.6 + similarity_score * 0.4
            confidence = product_confidence
        
        if final_score is not None:
            product_scores[product_id] = final_score
            product_confidence[product_id] = confidence
    
    # --- CÁCH TIẾP CẬN 3: Tăng precision bằng cách lọc theo ngưỡng độ tin cậy ---
    
    # Chỉ giữ lại các sản phẩm có điểm tin cậy đủ cao
    confidence_threshold = 0.2 if user_rating_count < 3 else 0.3  # Linh động dựa trên số lượng đánh giá
    filtered_products = [(pid, score) for pid, score in product_scores.items() 
                       if pid in product_confidence and product_confidence[pid] >= confidence_threshold]
    
    # Sắp xếp theo điểm số và chỉ lấy top N
    sorted_products = sorted(filtered_products, key=lambda x: x[1], reverse=True)[:top_n*2]
    
    # Chọn ra một tập đa dạng có điểm số cao nhất
    selected_products = []
    product_types_selected = set()
    
    # Ưu tiên chọn những sản phẩm có điểm cao nhất đầu tiên
    for product_id, score in sorted_products:
        product_row = product_details[product_details['product_id'] == product_id]
        
        if len(selected_products) < top_n:
            # Thêm sản phẩm vào danh sách đã chọn
            selected_products.append((product_id, score))
            
            # Theo dõi loại sản phẩm đã chọn để đảm bảo đa dạng
            if not product_row.empty and 'product_type' in product_row.columns:
                product_type = product_row['product_type'].iloc[0]
                product_types_selected.add(product_type)
        else:
            # Đã đạt đủ số lượng sản phẩm cần thiết
            break
    
    # Lấy thông tin chi tiết và thêm điểm cho các sản phẩm được chọn
    top_product_ids = [product_id for product_id, _ in selected_products]
    scores = {product_id: score for product_id, score in selected_products}
    
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    
    # Thêm điểm dự đoán vào kết quả
    recommended_products['predicted_rating'] = recommended_products['product_id'].map(scores)
    
    # Sắp xếp theo điểm giảm dần
    recommended_products = recommended_products.sort_values(by='predicted_rating', ascending=False)
    
    # Tính toán mức độ đa dạng của kết quả gợi ý
    if len(top_product_ids) >= 2 and all(pid in id_to_index for pid in top_product_ids):
        diversity_score = calculate_diversity(top_product_ids, id_to_index, similarity_matrix)
        print(f"Recommendation diversity: {diversity_score:.3f}")
    
    return recommended_products

class MockCollaborativeModel:
    def predict(self, uid, iid):
        return type('MockPrediction', (object,), {'est': 4.5})()

collaborative_model = MockCollaborativeModel()

# Implementation of the WeightedHybridRecommender class with proper indentation and methods
class WeightedHybridRecommender:
    """
    A class-based implementation of weighted hybrid recommender
    
    Attributes:
    -----------
    collaborative_model : object
        Trained collaborative filtering model
    content_model_data : tuple
        Tuple containing content-based model data
    weights : dict
        Dictionary of weights for different recommendation approaches
    """
    
    def __init__(self, collaborative_model, content_model_data, weights=None):
        """
        Initialize the weighted hybrid recommender
        
        Parameters:
        -----------
        collaborative_model : object
            Trained collaborative filtering model
        content_model_data : tuple
            Tuple containing content-based model data
        weights : dict, optional
            Dictionary of weights for different recommendation approaches
        """
        self.collaborative_model = collaborative_model
        self.content_model_data = content_model_data
        self.weights = weights or {'collaborative': 0.5, 'content': 0.3, 'event': 0.2}
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def recommend(self, user_id, product_details, events_df=None, top_n=10):
        """
        Generate weighted hybrid recommendations
        
        Parameters:
        -----------
        user_id : int
            User ID for which recommendations are to be generated
        product_details : DataFrame
            DataFrame containing product details
        events_df : DataFrame, optional
            DataFrame containing event data
        top_n : int, optional
            Number of recommendations to return
            
        Returns:
        --------
        DataFrame
            DataFrame containing recommended products with scores
        """
        return hybrid_recommendations(
            collaborative_model=self.collaborative_model,
            content_model_data=self.content_model_data,
            user_id=user_id,
            product_details=product_details,
            user_events=events_df,
            top_n=top_n
        )