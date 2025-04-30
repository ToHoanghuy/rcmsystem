import pandas as pd
import numpy as np
from surprise import accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Tính Precision và Recall ở ngưỡng K
    
    Parameters:
    -----------
    predictions: list
        List các dự đoán từ mô hình Surprise (uid, iid, true_r, est, details)
    k: int
        Số lượng sản phẩm được gợi ý để xem xét
    threshold: float
        Ngưỡng cho rating để xác định relevent/non-relevent
        
    Returns:
    --------
    precision: float
        Tỷ lệ sản phẩm được gợi ý mà có liên quan (true positive)
    recall: float
        Tỷ lệ sản phẩm có liên quan mà được gợi ý
    """
    # Nhóm các dự đoán theo người dùng
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    precisions = []
    recalls = []
    
    for uid, user_ratings in user_est_true.items():
        # Sắp xếp theo điểm dự đoán giảm dần
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Lấy top k sản phẩm
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = min(k, len(user_ratings))
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in user_ratings[:n_rec_k])
        
        # Tính precision@k và recall@k
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k > 0 else 1
        recall = n_rel_and_rec_k / n_rel if n_rel > 0 else 1
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Tính trung bình
    return np.mean(precisions), np.mean(recalls)

def ndcg_at_k(predictions, k=10):
    """
    Tính Normalized Discounted Cumulative Gain (NDCG) ở ngưỡng K
    
    Parameters:
    -----------
    predictions: list
        List các dự đoán từ mô hình Surprise (uid, iid, true_r, est, details)
    k: int
        Số lượng sản phẩm được gợi ý để xem xét
        
    Returns:
    --------
    ndcg: float
        Giá trị NDCG trung bình trên tất cả người dùng
    """
    # Nhóm các dự đoán theo người dùng
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    ndcg_scores = []
    
    for uid, user_ratings in user_est_true.items():
        # Sắp xếp theo điểm dự đoán giảm dần
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Tính DCG
        dcg = 0
        for i, (_, true_r) in enumerate(user_ratings[:k]):
            # Sử dụng log cơ số 2 cho DCG
            dcg += true_r / np.log2(i + 2)  # i+2 vì i bắt đầu từ 0
        
        # Tính IDCG (Ideal DCG)
        ideal_ratings = sorted([true_r for (_, true_r) in user_ratings], reverse=True)
        idcg = 0
        for i, true_r in enumerate(ideal_ratings[:k]):
            idcg += true_r / np.log2(i + 2)
        
        # Tính NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    # Tính trung bình
    return np.mean(ndcg_scores)

def evaluate_model(predictions, k=10, threshold=3.5):
    """
    Đánh giá mô hình với nhiều tiêu chí
    
    Parameters:
    -----------
    predictions: list
        List các dự đoán từ mô hình Surprise (uid, iid, true_r, est, details)
    k: int
        Số lượng sản phẩm được gợi ý để xem xét
    threshold: float
        Ngưỡng cho rating để xác định relevent/non-relevent
        
    Returns:
    --------
    metrics: dict
        Dictionary chứa các tiêu chí đánh giá
    """
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    precision, recall = precision_recall_at_k(predictions, k=k, threshold=threshold)
    ndcg = ndcg_at_k(predictions, k=k)
    
    # Tính F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'ndcg@k': ndcg
    }
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"F1@{k}: {f1:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    
    return metrics

def evaluate_all_models(models, testsets, product_details):
    """
    Đánh giá tất cả các mô hình và trả về báo cáo chi tiết
    
    Parameters:
    -----------
    models: dict
        Dictionary chứa các mô hình (collaborative, content_based, hybrid, event_based)
    testsets: dict
        Dictionary chứa các tập kiểm tra
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    report: dict
        Dictionary chứa báo cáo đánh giá cho tất cả các mô hình
    """
    collaborative_model = models.get('collaborative')
    content_model_data = models.get('content_based')
    hybrid_model = models.get('hybrid')
    event_model = models.get('event_based')
    
    collaborative_testset = testsets.get('collaborative')
    
    report = {
        'collaborative': {},
        'content_based': {},
        'hybrid': {},
        'event_based': {}
    }
    
    # Đánh giá Collaborative Filtering
    if collaborative_model and collaborative_testset:
        print("\n=== Đánh giá Collaborative Filtering ===")
        predictions = collaborative_model.test(collaborative_testset)
        metrics = evaluate_model(predictions, k=10)
        report['collaborative'] = metrics
    
    # Đánh giá các mô hình khác tương tự
    # Nên thực hiện đánh giá offline với các tập dữ liệu thích hợp
    
    return report

def analyze_errors(collaborative_model, testset, product_details):
    """
    Phân tích lỗi dự đoán của collaborative model
    
    Parameters:
    -----------
    collaborative_model: surprise.SVD
        Mô hình Collaborative Filtering đã huấn luyện
    testset: list
        Tập kiểm tra dùng để đánh giá mô hình
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    error_analysis: dict
        Dictionary chứa các phân tích lỗi
    """
    predictions = collaborative_model.test(testset)
    
    # Tính lỗi cho từng dự đoán
    errors = []
    for uid, iid, true_r, est, _ in predictions:
        error = abs(true_r - est)
        errors.append((uid, iid, true_r, est, error))
    
    # Chuyển thành DataFrame
    error_df = pd.DataFrame(errors, columns=['user_id', 'product_id', 'true_rating', 'predicted_rating', 'error'])
    
    # Lấy thông tin sản phẩm và kết hợp
    product_lookup = {}
    for _, row in product_details.iterrows():
        product_lookup[row['product_id']] = {col: row[col] for col in product_details.columns}
    
    # Thêm thông tin sản phẩm vào error_df
    product_types = []
    product_prices = []
    product_ratings = []
    
    for _, row in error_df.iterrows():
        product_id = row['product_id']
        if product_id in product_lookup:
            product_types.append(product_lookup[product_id].get('product_type', 'unknown'))
            product_prices.append(product_lookup[product_id].get('price', 0))
            product_ratings.append(product_lookup[product_id].get('rating', 0))
        else:
            product_types.append('unknown')
            product_prices.append(0)
            product_ratings.append(0)
    
    error_df['product_type'] = product_types
    error_df['price'] = product_prices
    error_df['rating'] = product_ratings
    
    # Phân tích lỗi theo loại sản phẩm
    error_by_type = error_df.groupby('product_type')['error'].mean().sort_values(ascending=False)
    
    # Phân tích lỗi theo rating thực tế
    error_by_rating = error_df.groupby('true_rating')['error'].mean().sort_values(ascending=False)
    
    # Phân tích 10 sản phẩm có lỗi cao nhất
    top_errors = error_df.sort_values(by='error', ascending=False).head(10)
    
    # Phân tích lỗi theo khoảng giá
    error_df['price_range'] = pd.cut(error_df['price'], bins=[0, 100, 300, 500, 1000, float('inf')], 
                                   labels=['0-100', '100-300', '300-500', '500-1000', '1000+'])
    error_by_price = error_df.groupby('price_range')['error'].mean().sort_values(ascending=False)
    
    # Tạo visualizations nếu cần
    
    return {
        'mean_error': error_df['error'].mean(),
        'median_error': error_df['error'].median(),
        'error_by_type': error_by_type.to_dict(),
        'error_by_rating': error_by_rating.to_dict(),
        'error_by_price': error_by_price.to_dict(),
        'top_errors': top_errors.to_dict(orient='records')
    }

def plot_error_analysis(error_analysis, save_path=None):
    """
    Tạo biểu đồ phân tích lỗi
    
    Parameters:
    -----------
    error_analysis: dict
        Kết quả từ hàm analyze_errors
    save_path: str, optional
        Đường dẫn để lưu biểu đồ
        
    Returns:
    --------
    None
    """
    # Tạo figure với 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot lỗi theo loại sản phẩm
    error_by_type = pd.Series(error_analysis['error_by_type'])
    error_by_type.plot(kind='bar', ax=axs[0, 0], color='skyblue')
    axs[0, 0].set_title('Mean Error by Product Type')
    axs[0, 0].set_ylabel('Mean Absolute Error')
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot lỗi theo rating thực tế
    error_by_rating = pd.Series(error_analysis['error_by_rating'])
    error_by_rating.plot(kind='bar', ax=axs[0, 1], color='lightgreen')
    axs[0, 1].set_title('Mean Error by True Rating')
    axs[0, 1].set_ylabel('Mean Absolute Error')
    
    # Plot lỗi theo khoảng giá
    error_by_price = pd.Series(error_analysis['error_by_price'])
    error_by_price.plot(kind='bar', ax=axs[1, 0], color='salmon')
    axs[1, 0].set_title('Mean Error by Price Range')
    axs[1, 0].set_ylabel('Mean Absolute Error')
    
    # Plot histogram của lỗi
    top_errors = pd.DataFrame(error_analysis['top_errors'])
    if not top_errors.empty:
        axs[1, 1].hist(top_errors['error'], bins=20, color='purple', alpha=0.7)
        axs[1, 1].set_title('Distribution of Top Errors')
        axs[1, 1].set_xlabel('Error')
        axs[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Error analysis plot saved to {save_path}")
    else:
        plt.show()

def create_confusion_matrix(predictions, threshold=3.5):
    """
    Tạo confusion matrix cho các dự đoán
    
    Parameters:
    -----------
    predictions: list
        List các dự đoán từ mô hình Surprise (uid, iid, true_r, est, details)
    threshold: float
        Ngưỡng cho rating để xác định positive/negative
        
    Returns:
    --------
    cm: numpy.ndarray
        Confusion matrix
    """
    y_true = []
    y_pred = []
    
    for _, _, true_r, est, _ in predictions:
        y_true.append(1 if true_r >= threshold else 0)
        y_pred.append(1 if est >= threshold else 0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return cm, precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """
    Vẽ confusion matrix
    
    Parameters:
    -----------
    cm: numpy.ndarray
        Confusion matrix
    title: str
        Tiêu đề của biểu đồ
    save_path: str, optional
        Đường dẫn để lưu biểu đồ
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Rating')
    plt.xlabel('Predicted Rating')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()