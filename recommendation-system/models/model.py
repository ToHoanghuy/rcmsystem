from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

def train_model(data):
    # Kiểm tra dữ liệu đầu vào đủ đa dạng
    if data['user_id'].nunique() < 2 or data['product_id'].nunique() < 2:
        print("[CẢNH BÁO] Dữ liệu quá ít user hoặc sản phẩm để huấn luyện mô hình.")
        return None
    if data['rating'].nunique() < 2:
        print("[CẢNH BÁO] Tất cả rating đều giống nhau, không thể huấn luyện mô hình phân biệt.")
        return None
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    
    # Tối ưu tham số mô hình
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.05]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(dataset)
    
    algo = gs.best_estimator['rmse']
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    # Đánh giá mô hình
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(f"Best RMSE: {gs.best_score['rmse']}")
    print(f"Best MAE: {gs.best_score['mae']}")
    print(f"Best Params: {gs.best_params['rmse']}")
    return algo