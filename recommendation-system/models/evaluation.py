import pandas as pd
import numpy as np
from surprise import accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from collections import defaultdict

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

def mean_reciprocal_rank(predictions):
    """
    Calculate Mean Reciprocal Rank (MRR) for recommendations
    
    Parameters:
    -----------
    predictions: list
        List of predictions from Surprise model (uid, iid, true_r, est, details)
        
    Returns:
    --------
    mrr: float
        Mean Reciprocal Rank score
    """
    # Group predictions by user
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((iid, est, true_r))
    
    reciprocal_ranks = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by predicted rating (highest first)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Find the first relevant item
        for rank, (_, _, true_r) in enumerate(user_ratings, 1):
            if true_r >= 4.0:  # Consider 4+ as highly relevant
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant items found
            reciprocal_ranks.append(0.0)
    
    # Calculate mean
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def coverage(predictions, catalog_size):
    """
    Calculate recommendation coverage (% of available items that are recommended)
    
    Parameters:
    -----------
    predictions: list
        List of predictions from Surprise model
    catalog_size: int
        Total number of items in the catalog
        
    Returns:
    --------
    coverage: float
        Coverage percentage (0-100)
    """
    recommended_items = set()
    for _, iid, _, _, _ in predictions:
        recommended_items.add(iid)
    
    return (len(recommended_items) / catalog_size) * 100

def diversity(predictions, item_features):
    """
    Calculate diversity of recommendations based on item features
    
    Parameters:
    -----------
    predictions: list
        List of predictions from Surprise model
    item_features: DataFrame
        DataFrame containing item features
        
    Returns:
    --------
    diversity: float
        Average pairwise distance between recommended items (higher = more diverse)
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_predictions[uid].append((iid, est))
    
    diversity_scores = []
    
    for uid, items in user_predictions.items():
        # Sort by prediction score and take top 10
        items.sort(key=lambda x: x[1], reverse=True)
        top_items = items[:10]
        
        # Calculate pairwise distances between items
        if len(top_items) < 2:
            continue
            
        pair_diversities = []
        for i in range(len(top_items)):
            for j in range(i+1, len(top_items)):
                item_i = top_items[i][0]
                item_j = top_items[j][0]
                
                # Skip if items not in the features dataframe
                if item_i not in item_features.index or item_j not in item_features.index:
                    continue
                
                # Calculate feature distance (using category or other feature columns)
                features_i = item_features.loc[item_i]
                features_j = item_features.loc[item_j]
                
                # Use categorical features to compute Jaccard distance
                categorical_cols = item_features.select_dtypes(include=['object', 'category']).columns
                
                if len(categorical_cols) > 0:
                    common_categories = sum(features_i[col] == features_j[col] for col in categorical_cols)
                    jaccard_dist = 1 - (common_categories / len(categorical_cols))
                    pair_diversities.append(jaccard_dist)
        
        if pair_diversities:
            diversity_scores.append(np.mean(pair_diversities))
    
    return np.mean(diversity_scores) if diversity_scores else 0.0

def novelty(predictions, popularity_df):
    """
    Calculate novelty of recommendations (inverse of popularity)
    
    Parameters:
    -----------
    predictions: list
        List of predictions from Surprise model
    popularity_df: DataFrame
        DataFrame with item popularity information
        
    Returns:
    --------
    novelty: float
        Average novelty score (higher = more novel/less popular items)
    """
    # Group predictions by user
    user_predictions = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_predictions[uid].append((iid, est))
    
    novelty_scores = []
    
    for uid, items in user_predictions.items():
        # Sort by prediction score and take top k
        items.sort(key=lambda x: x[1], reverse=True)
        top_items = items[:10]
        
        # Calculate novelty for each item (inverse of popularity)
        item_novelties = []
        for iid, _ in top_items:
            if iid in popularity_df.index:
                pop = popularity_df.loc[iid, 'popularity']
                # Convert popularity to novelty (inverse relationship)
                novelty = 1.0 - (pop / popularity_df['popularity'].max())
                item_novelties.append(novelty)
        
        if item_novelties:
            novelty_scores.append(np.mean(item_novelties))
    
    return np.mean(novelty_scores) if novelty_scores else 0.0

def evaluate_model(predictions, k=10, threshold=3.5, catalog_size=None, item_features=None, popularity_df=None):
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
    catalog_size: int, optional
        Total number of items in the catalog (for coverage calculation)
    item_features: DataFrame, optional
        DataFrame with item features (for diversity calculation)
    popularity_df: DataFrame, optional
        DataFrame with item popularity information (for novelty calculation)
        
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
    
    # Calculate MRR
    mrr = mean_reciprocal_rank(predictions)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'ndcg@k': ndcg,
        'mrr': mrr
    }
    
    # Add advanced metrics if data is provided
    if catalog_size:
        cov = coverage(predictions, catalog_size)
        metrics['coverage'] = cov
        print(f"Coverage: {cov:.2f}%")
    
    if item_features is not None:
        div = diversity(predictions, item_features)
        metrics['diversity'] = div
        print(f"Diversity: {div:.4f}")
    
    if popularity_df is not None:
        nov = novelty(predictions, popularity_df)
        metrics['novelty'] = nov
        print(f"Novelty: {nov:.4f}")
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"F1@{k}: {f1:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    return metrics

def evaluate_all_models(models, testsets, product_details, catalog_size=None, popularity_df=None):
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
    catalog_size: int, optional
        Total number of items in the catalog
    popularity_df: DataFrame, optional
        DataFrame with item popularity information
        
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
    
    # Create item features DataFrame for diversity calculation
    item_features = None
    if 'product_type' in product_details.columns or 'category' in product_details.columns:
        # Use product type or category as features
        feature_cols = [col for col in product_details.columns 
                       if col in ['product_type', 'category', 'subcategory', 'brand']]
        if feature_cols:
            item_features = product_details[['product_id'] + feature_cols].set_index('product_id')
    
    # Đánh giá Collaborative Filtering
    if collaborative_model and collaborative_testset:
        print("\n=== Đánh giá Collaborative Filtering ===")
        predictions = collaborative_model.test(collaborative_testset)
        metrics = evaluate_model(predictions, k=10, catalog_size=catalog_size, 
                               item_features=item_features, popularity_df=popularity_df)
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
            
            # Đảm bảo price là giá trị số hợp lệ
            try:
                price = float(product_lookup[product_id].get('price', 0))
                if np.isnan(price):
                    price = 0
            except (ValueError, TypeError):
                price = 0
            product_prices.append(price)
            
            # Đảm bảo rating là giá trị số hợp lệ
            try:
                rating = float(product_lookup[product_id].get('rating', 0))
                if np.isnan(rating):
                    rating = 0
            except (ValueError, TypeError):
                rating = 0
            product_ratings.append(rating)
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
    
    # Phân tích lỗi theo khoảng giá - cải tiến cách xử lý khoảng giá
    # Điều chỉnh khoảng giá dựa trên phân phối thực tế của dữ liệu
    price_stats = error_df['price'].describe()
    
    # Xác định khoảng giá dựa trên phân vị thay vì giá trị cố định
    q25, q50, q75 = price_stats['25%'], price_stats['50%'], price_stats['75%']
    max_price = price_stats['max']
    
    # Điều chỉnh khoảng giá để đảm bảo mỗi khoảng có dữ liệu
    if max_price > 0:
        # Tính toán các ngưỡng khoảng giá linh động
        bins = [0, max(q25, 100), max(q50, 300), max(q75, 500), max_price]
        labels = [f'0-{int(bins[1])}', f'{int(bins[1])}-{int(bins[2])}', 
                 f'{int(bins[2])}-{int(bins[3])}', f'{int(bins[3])}+']
        
        # Tạo cột price_range với các ngưỡng được điều chỉnh
        error_df['price_range'] = pd.cut(
            error_df['price'], 
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Tính lỗi trung bình theo price_range
        error_by_price = error_df.groupby('price_range')['error'].mean()
        
        # Đảm bảo không có giá trị NaN bằng cách điền 0 cho khoảng không có dữ liệu
        for label in labels:
            if label not in error_by_price.index:
                error_by_price[label] = 0
    else:
        # Nếu không có giá trị price hợp lệ, tạo Series giả
        error_by_price = pd.Series({
            '0-100': 0,
            '100-300': 0,
            '300-500': 0,
            '500+': 0
        })
    
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

def compare_models_radar_chart(evaluation_results, save_path=None):
    """
    Create a radar chart to compare different recommendation models
    
    Parameters:
    -----------
    evaluation_results: dict
        Dictionary with evaluation results for different models
    save_path: str, optional
        Path to save the chart image
    
    Returns:
    --------
    None
    """
    # Metrics to compare (normalize metrics for fair comparison)
    metrics = ['precision@k', 'recall@k', 'ndcg@k', 'mrr']
    if 'diversity' in list(evaluation_results.values())[0]:
        metrics.append('diversity')
    if 'novelty' in list(evaluation_results.values())[0]:
        metrics.append('novelty')
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add metric labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Plot each model
    for model_name, results in evaluation_results.items():
        values = [results.get(metric, 0) for metric in metrics]
        
        # Normalize values to [0,1] for fair comparison
        min_vals = {'precision@k': 0, 'recall@k': 0, 'ndcg@k': 0, 'mrr': 0, 'diversity': 0, 'novelty': 0}
        max_vals = {'precision@k': 1, 'recall@k': 1, 'ndcg@k': 1, 'mrr': 1, 'diversity': 1, 'novelty': 1}
        
        norm_values = [(v - min_vals.get(metrics[i], 0)) / (max_vals.get(metrics[i], 1) - min_vals.get(metrics[i], 0)) 
                      for i, v in enumerate(values)]
        
        # Complete the loop
        norm_values += norm_values[:1]
        
        # Plot values
        ax.plot(angles, norm_values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, norm_values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison', size=15)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Radar chart saved to {save_path}")
    else:
        plt.show()