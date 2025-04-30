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
        Nếu True, thực hiện tối ưu hóa tham số với GridSearchCV
        
    Returns:
    --------
    model: SVD model
        Mô hình đã được huấn luyện
    testset: list
        Tập dữ liệu kiểm tra
    predictions: list
        Các dự đoán trên tập kiểm tra
    """
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    
    if optimize_params:
        print("Optimizing collaborative filtering parameters...")
        # Tối ưu tham số mô hình bằng GridSearchCV
        param_grid = {
            'n_factors': [50, 100],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.05]
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(dataset)
        
        # Sử dụng mô hình tốt nhất
        model = gs.best_estimator['rmse']
        print(f"Best parameters: {gs.best_params['rmse']}")
        print(f"Best RMSE: {gs.best_score['rmse']}")
        print(f"Best MAE: {gs.best_score['mae']}")
    else:
        # Sử dụng tham số mặc định
        model = SVD()
    
    # Huấn luyện mô hình trên tập huấn luyện
    model.fit(trainset)
    
    # Đánh giá mô hình
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    
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
    user_id: int
        ID của người dùng
    product_id: int
        ID của sản phẩm
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
    try:
        # Thử dự đoán bằng CF
        prediction = model.predict(uid=user_id, iid=product_id)
        return prediction.est
    except:
        # Nếu không được, sử dụng content-based
        try:
            # Tìm index của sản phẩm trong product_details
            product_idx = product_details[product_details['product_id'] == product_id].index[0]
            
            # Lấy top 5 sản phẩm tương tự
            similar_indices = content_similarity[product_idx].argsort()[-5:][::-1]
            similar_ratings = []
            
            # Lấy điểm đánh giá của người dùng cho các sản phẩm tương tự
            for idx in similar_indices:
                similar_product_id = product_details.iloc[idx]['product_id']
                try:
                    # Thử dự đoán điểm đánh giá
                    pred = model.predict(uid=user_id, iid=similar_product_id)
                    similar_ratings.append(pred.est)
                except:
                    pass
            
            # Nếu có đánh giá từ các sản phẩm tương tự, trả về điểm trung bình
            if similar_ratings:
                return sum(similar_ratings) / len(similar_ratings)
            else:
                return 3.0  # Giá trị mặc định
        except Exception as e:
            print(f"Cold-start prediction error: {e}")
            return 3.0  # Giá trị mặc định