import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_in_batches(data, batch_size=1000, process_func=None):
    """
    Xử lý dữ liệu lớn theo batch để giảm sử dụng bộ nhớ
    
    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame cần xử lý
    batch_size: int
        Kích thước của mỗi batch
    process_func: function, optional
        Hàm xử lý cho mỗi batch
        
    Returns:
    --------
    result: pandas.DataFrame or list
        Kết quả sau khi xử lý
    """
    result = []
    
    # In tiến trình xử lý
    print(f"Processing data in batches of {batch_size}...")
    
    for start_idx in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(data))
        batch = data.iloc[start_idx:end_idx]
        
        if process_func:
            # Áp dụng hàm xử lý cho batch
            processed_batch = process_func(batch)
            result.append(processed_batch)
        else:
            result.append(batch)
    
    # Kết hợp kết quả
    if isinstance(result[0], pd.DataFrame):
        return pd.concat(result)
    else:
        # Nếu kết quả không phải DataFrame, ghép chúng lại
        combined_result = []
        for r in result:
            if isinstance(r, list):
                combined_result.extend(r)
            else:
                combined_result.append(r)
        return combined_result

def parallel_process_batches(data, batch_size=1000, process_func=None, n_workers=None):
    """
    Xử lý dữ liệu lớn theo batch với tính toán song song
    
    Parameters:
    -----------
    data: pandas.DataFrame
        DataFrame cần xử lý
    batch_size: int
        Kích thước của mỗi batch
    process_func: function
        Hàm xử lý cho mỗi batch
    n_workers: int, optional
        Số lượng worker, mặc định là số lõi CPU - 1
        
    Returns:
    --------
    result: pandas.DataFrame or list
        Kết quả sau khi xử lý
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Chia dữ liệu thành các batch
    batches = []
    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        batches.append(data.iloc[start_idx:end_idx])
    
    # Xử lý các batch song song
    print(f"Processing data with {n_workers} workers...")
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.map(process_func, batches), total=len(batches), desc="Processing batches"))
    
    # Kết hợp kết quả
    if isinstance(results[0], pd.DataFrame):
        return pd.concat(results)
    else:
        # Nếu kết quả không phải DataFrame, ghép chúng lại
        combined_result = []
        for r in results:
            if isinstance(r, list):
                combined_result.extend(r)
            else:
                combined_result.append(r)
        return combined_result

def process_large_similarity_matrix(product_ids, feature_func, batch_size=100):
    """
    Tính ma trận tương đồng cho dữ liệu lớn theo batch
    
    Parameters:
    -----------
    product_ids: list
        Danh sách product_id để tính toán
    feature_func: function
        Hàm trích xuất đặc trưng cho mỗi product_id
    batch_size: int
        Kích thước của mỗi batch
        
    Returns:
    --------
    similarity_matrix: numpy.ndarray
        Ma trận tương đồng
    """
    from sklearn.metrics.pairwise import cosine_similarity
    n_products = len(product_ids)
    similarity_matrix = np.zeros((n_products, n_products))
    
    print(f"Computing similarity matrix for {n_products} products in batches of {batch_size}...")
    
    # Tính toán ma trận tương đồng theo batch
    for i in tqdm(range(0, n_products, batch_size), desc="Computing similarity"):
        i_end = min(i + batch_size, n_products)
        
        # Lấy đặc trưng cho batch hiện tại
        batch_i_features = np.vstack([feature_func(pid) for pid in product_ids[i:i_end]])
        
        for j in range(0, n_products, batch_size):
            j_end = min(j + batch_size, n_products)
            
            # Lấy đặc trưng cho batch so sánh
            if j == i:
                batch_j_features = batch_i_features
            else:
                batch_j_features = np.vstack([feature_func(pid) for pid in product_ids[j:j_end]])
            
            # Tính toán tương đồng giữa hai batch
            batch_similarity = cosine_similarity(batch_i_features, batch_j_features)
            
            # Cập nhật ma trận tương đồng
            similarity_matrix[i:i_end, j:j_end] = batch_similarity
    
    return similarity_matrix

def batch_predict(model, user_ids, product_ids, batch_size=1000):
    """
    Dự đoán rating cho nhiều user và product với xử lý batch
    
    Parameters:
    -----------
    model: surprise.AlgoBase
        Mô hình dự đoán
    user_ids: list
        Danh sách user_id
    product_ids: list
        Danh sách product_id
    batch_size: int
        Kích thước của mỗi batch
        
    Returns:
    --------
    predictions: dict
        Dictionary chứa prediction cho mỗi cặp (user_id, product_id)
    """
    predictions = {}
    total_pairs = len(user_ids) * len(product_ids)
    
    print(f"Predicting ratings for {len(user_ids)} users and {len(product_ids)} products...")
    
    # Tạo tất cả các cặp (user_id, product_id)
    pairs = []
    for uid in user_ids:
        for pid in product_ids:
            pairs.append((uid, pid))
    
    # Xử lý các cặp theo batch
    with tqdm(total=len(pairs), desc="Predicting ratings") as pbar:
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_predictions = []
            
            for uid, pid in batch_pairs:
                try:
                    pred = model.predict(uid, pid)
                    batch_predictions.append((uid, pid, pred.est))
                except Exception as e:
                    # Bỏ qua nếu không dự đoán được
                    pass
            
            # Cập nhật dictionary kết quả
            for uid, pid, est in batch_predictions:
                if uid not in predictions:
                    predictions[uid] = {}
                predictions[uid][pid] = est
            
            pbar.update(len(batch_pairs))
    
    return predictions