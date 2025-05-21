from surprise import Dataset, Reader, SVD, SVDpp
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import accuracy
import pandas as pd

def train_collaborative_model(data, optimize_params=True):
    """
    Huấn luyện mô hình Collaborative Filtering
    
    Parameters:
    -----------
    data: DataFrame
        DataFrame chứa dữ liệu đánh giá với các cột 'user_id', 'product_id', 'rating'
    optimize_params: bool
        Có tối ưu tham số mô hình hay không
        
    Returns:
    --------
    model: SVD model
        Mô hình đã được huấn luyện
    testset: list
        Tập dữ liệu kiểm tra
    predictions: list
        Các dự đoán trên tập kiểm tra
    """    # Check and handle different column naming conventions
    # Look for 'rating' column or alternatives like 'rating_x' or 'integrated_score'
    rating_col = 'rating'
    if rating_col not in data.columns:
        if 'rating_x' in data.columns:
            rating_col = 'rating_x'
            print(f"Using '{rating_col}' column instead of 'rating'")
        elif 'integrated_score' in data.columns:
            rating_col = 'integrated_score'
            print(f"Using '{rating_col}' column instead of 'rating'")
        else:
            raise ValueError("No suitable rating column found in the data")
    
    # Cân bằng dữ liệu đánh giá trước khi huấn luyện mô hình
    # Xử lý vấn đề mất cân bằng giữa các rating cực trị (1 và 5)
    rating_counts = data[rating_col].value_counts().sort_index()
    print(f"Original rating distribution: {rating_counts.to_dict()}")
    
    # Cải thiện: Nếu có quá nhiều đánh giá cực trị, thực hiện lấy mẫu để cân bằng dữ liệu
    if 5.0 in rating_counts.index and 1.0 in rating_counts.index:
        max_rating_count = rating_counts[5.0]
        min_rating_count = rating_counts[1.0]
        # Nếu số lượng rating 5.0 nhiều hơn 3 lần rating 1.0
        if max_rating_count > 3 * min_rating_count and min_rating_count > 0:
            # Lấy mẫu ngẫu nhiên từ các đánh giá 5.0
            rating_5_samples = data[data[rating_col] == 5.0].sample(n=min(max_rating_count, 3 * min_rating_count), random_state=42)
            # Kết hợp với các đánh giá khác
            other_ratings = data[data[rating_col] != 5.0]
            data = pd.concat([other_ratings, rating_5_samples])
            print(f"Balanced high rating samples to reduce bias. New count: {len(data[data[rating_col] == 5.0])}")
    reader = Reader(rating_scale=(1, 5))
    # Create a copy of the DataFrame with standard column names for Surprise
    surprise_data = data.copy()
    surprise_data['rating'] = surprise_data[rating_col]
    
    # Đảm bảo user_id và product_id có thể sử dụng với Surprise
    # Surprise yêu cầu các ID phải là số hoặc chuỗi, nhưng thường thích số hơn
    try:
        # Nếu là chuỗi, thử chuyển sang số nguyên
        if surprise_data['user_id'].dtype == object:
            surprise_data['user_id'] = pd.to_numeric(surprise_data['user_id'], errors='ignore')
        if surprise_data['product_id'].dtype == object:
            surprise_data['product_id'] = pd.to_numeric(surprise_data['product_id'], errors='ignore')
    except Exception as e:
        print(f"Warning: Could not convert IDs to numeric, will use as is: {e}")
    
    print(f"Data types: user_id: {surprise_data['user_id'].dtype}, product_id: {surprise_data['product_id'].dtype}")
    dataset = Dataset.load_from_df(surprise_data[['user_id', 'product_id', 'rating']], reader)
    
    # Thêm trọng số ngược với tần suất để tăng ảnh hưởng của rating hiếm
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Cải thiện: Thêm thông báo về kích thước tập dữ liệu
    print(f"Training set size: {trainset.n_ratings} ratings")
    print(f"Test set size: {len(testset)} ratings")
    
    if optimize_params:
        print("Optimizing collaborative filtering parameters...")
        # Tối ưu tham số mô hình bằng GridSearchCV với tham số tốt hơn
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [20, 30, 50],
            'lr_all': [0.005, 0.007, 0.01],
            'reg_all': [0.02, 0.05, 0.1],
            'biased': [True]  # Thêm bias vào mô hình để xử lý tốt hơn các rating cực trị
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(dataset)
        
        # Sử dụng mô hình tốt nhất
        model = gs.best_estimator['rmse']
        print(f"Best parameters: {gs.best_params['rmse']}")
        print(f"Best RMSE: {gs.best_score['rmse']}")
        print(f"Best MAE: {gs.best_score['mae']}")
    else:
        # Sử dụng tham số mặc định cải tiến
        model = SVD(n_factors=100, n_epochs=30, biased=True, lr_all=0.007, reg_all=0.05)
    
    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(trainset)
    
    # Đánh giá mô hình
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    
    # Phân tích lỗi dự đoán theo từng mức rating
    error_by_rating = {}
    for _, _, true_r, est, _ in predictions:
        error = abs(true_r - est)
        rating = int(true_r) if true_r.is_integer() else true_r
        if rating not in error_by_rating:
            error_by_rating[rating] = []
        error_by_rating[rating].append(error)
    
    # In ra lỗi trung bình theo từng mức rating
    print("Error by rating level:")
    for rating, errors in sorted(error_by_rating.items()):
        mean_error = sum(errors) / len(errors)
        print(f"  Rating {rating}: MAE = {mean_error:.4f} (count: {len(errors)})")
    
    return model, testset, predictions

def train_advanced_collaborative_model(data):
    """
    Huấn luyện mô hình Collaborative Filtering nâng cao (SVD++) để xử lý dữ liệu thưa
    
    Parameters:
    -----------
    data: DataFrame
        DataFrame chứa dữ liệu đánh giá với các cột 'user_id', 'product_id', 'rating'
        
    Returns:
    --------
    model: SVDpp model
        Mô hình đã được huấn luyện
    testset: list
        Tập dữ liệu kiểm tra
    predictions: list
        Các dự đoán trên tập kiểm tra
    """
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    
    # Sử dụng SVD++ (tích hợp thông tin ngầm)
    print("Training SVD++ model...")
    model = SVDpp(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    
    # Đánh giá mô hình
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"SVD++ RMSE: {rmse}")
    print(f"SVD++ MAE: {mae}")
    
    return model, testset, predictions

def predict_for_cold_start(user_id, product_id, model, content_similarity, product_details):
    """
    Dự đoán đánh giá cho người dùng mới hoặc sản phẩm mới (cold-start problem)
    
    Parameters:
    -----------
    user_id: int or str
        ID của người dùng (có thể là chuỗi cho MongoDB IDs)
    product_id: int or str
        ID của sản phẩm (có thể là chuỗi cho MongoDB IDs)
    model: SVD/SVDpp model
        Mô hình Collaborative Filtering
    content_similarity: numpy.ndarray
        Ma trận tương đồng từ Content-Based Filtering
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về các sản phẩm
        
    Returns:
    --------
    float
        Điểm đánh giá dự đoán
    """
    # Khởi tạo mức độ tự tin về dự đoán
    confidence = 0.0
    
    try:
        # Thử dự đoán bằng CF, đảm bảo kiểu dữ liệu đúng với model
        # Surprise có thể yêu cầu uid và iid là chuỗi hoặc số nguyên
        try:
            prediction = model.predict(uid=user_id, iid=product_id)
        except (ValueError, TypeError) as e:
            # Nếu là lỗi chuyển đổi kiểu dữ liệu, thử với dạng chuỗi
            print(f"Trying prediction with string IDs: {e}")
            prediction = model.predict(uid=str(user_id), iid=str(product_id))
            
        confidence = 1.0  # Độ tin cậy cao nhất khi có dự đoán trực tiếp
        estimated_rating = prediction.est
    except Exception as e:
        print(f"Collaborative prediction failed: {e}")
        # Nếu không được, sử dụng content-based
        try:
            # Tìm index của sản phẩm trong product_details, đảm bảo kiểu dữ liệu phù hợp
            if isinstance(product_id, (int, float, str)):
                product_row = product_details[product_details['product_id'].astype(str) == str(product_id)]
            else:
                product_row = product_details[product_details['product_id'] == product_id]
            
            if product_row.empty:
                # Sản phẩm không tồn tại trong dữ liệu
                return 3.0  # Giá trị mặc định trung bình
            
            product_idx = product_row.index[0]
            
            # Lấy top 5 sản phẩm tương tự
            similar_indices = content_similarity[product_idx].argsort()[-10:][::-1]
            similar_ratings = []
            similarity_weights = []
            
            # Lấy điểm đánh giá của người dùng cho các sản phẩm tương tự
            for idx in similar_indices:
                if idx >= len(product_details):
                    continue
                
                similar_product_id = product_details.iloc[idx]['product_id']
                similarity = content_similarity[product_idx, idx]
                
                # Chỉ xem xét các sản phẩm đủ tương tự
                if similarity < 0.2:
                    continue
                
                try:
                    # Thử lấy đánh giá thực tế của người dùng cho sản phẩm tương tự
                    # Thử với cả kiểu dữ liệu gốc và kiểu string cho user_id
                    try:
                        inner_uid = model.trainset.to_inner_uid(user_id)
                    except (ValueError, KeyError):
                        try:
                            inner_uid = model.trainset.to_inner_uid(str(user_id))
                        except:
                            raise ValueError(f"User ID {user_id} not found in trainset")
                    
                    # Thử với cả kiểu dữ liệu gốc và kiểu string cho product_id
                    try:
                        inner_iid = model.trainset.to_inner_iid(similar_product_id)
                    except (ValueError, KeyError):
                        try:
                            inner_iid = model.trainset.to_inner_iid(str(similar_product_id))
                        except:
                            raise ValueError(f"Similar product ID {similar_product_id} not found in trainset")
                    
                    user_ratings = model.trainset.ur[inner_uid]
                    
                    for item_idx, rating_val in user_ratings:
                        if item_idx == inner_iid:
                            # Người dùng đã đánh giá sản phẩm tương tự này
                            similar_ratings.append(rating_val)
                            similarity_weights.append(similarity)
                            break
                except:
                    # Nếu không có đánh giá thực tế, thử dự đoán
                    try:
                        # Thử dự đoán với kiểu dữ liệu gốc
                        try:
                            pred = model.predict(uid=user_id, iid=similar_product_id)
                        except:
                            # Nếu có lỗi, thử với chuỗi
                            pred = model.predict(uid=str(user_id), iid=str(similar_product_id))
                        
                        similar_ratings.append(pred.est)
                        similarity_weights.append(similarity * 0.8)  # Giảm trọng số cho dự đoán
                    except:
                        pass
            
            # Nếu có đánh giá từ các sản phẩm tương tự, tính trung bình có trọng số
            if similar_ratings:
                # Tính độ tin cậy dựa trên số lượng và chất lượng các đánh giá tương tự
                confidence = min(0.8, len(similar_ratings) / 10.0)
                
                # Tính trung bình có trọng số
                if sum(similarity_weights) > 0:
                    estimated_rating = sum(r * w for r, w in zip(similar_ratings, similarity_weights)) / sum(similarity_weights)
                else:
                    estimated_rating = sum(similar_ratings) / len(similar_ratings)
                
                # Kiểm tra dữ liệu sản phẩm
                if 'rating' in product_row.columns and not pd.isna(product_row['rating'].iloc[0]):
                    # Kết hợp với rating trung bình của sản phẩm (nếu có)
                    avg_product_rating = product_row['rating'].iloc[0]
                    # Trọng số cho rating trung bình phụ thuộc vào độ tin cậy của các dự đoán tương tự
                    product_weight = 1.0 - confidence
                    estimated_rating = (confidence * estimated_rating + product_weight * avg_product_rating) / (confidence + product_weight)
            else:
                # Không có sản phẩm tương tự, sử dụng rating trung bình của sản phẩm nếu có
                if 'rating' in product_row.columns and not pd.isna(product_row['rating'].iloc[0]):
                    estimated_rating = product_row['rating'].iloc[0]
                    confidence = 0.4  # Độ tin cậy trung bình
                else:
                    # Không có thông tin nào, sử dụng giá trị mặc định
                    estimated_rating = 3.0
                    confidence = 0.1  # Độ tin cậy thấp
        
        except Exception as e:
            print(f"Cold-start prediction error: {e}")
            estimated_rating = 3.0  # Giá trị mặc định
            confidence = 0.0  # Không có độ tin cậy
    
    # Điều chỉnh các dự đoán cực trị để giảm thiểu lỗi lớn
    # Các dự đoán gần 1.0 hoặc 5.0 có thể bị điều chỉnh nhẹ về phía trung tâm
    if estimated_rating < 1.5 and confidence < 0.9:
        # Giảm thiểu việc đánh giá quá thấp khi độ tin cậy không cao
        estimated_rating = 1.5 + (1.0 - confidence) * 0.5
    elif estimated_rating > 4.5 and confidence < 0.9:
        # Giảm thiểu việc đánh giá quá cao khi độ tin cậy không cao
        estimated_rating = 4.5 - (1.0 - confidence) * 0.5
    
    return estimated_rating