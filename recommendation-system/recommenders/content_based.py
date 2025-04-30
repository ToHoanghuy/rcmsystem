from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def extract_features(product_details):
    """
    Trích xuất và kết hợp các đặc trưng khác nhau từ dữ liệu sản phẩm
    
    Parameters:
    -----------
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    all_features: DataFrame
        DataFrame chứa tất cả các đặc trưng đã xử lý
    """
    # Lưu lại product_id để sau này map lại
    product_ids = product_details['product_id'].tolist()
    
    # Xử lý đặc trưng phân loại với one-hot encoding
    cat_columns = []
    for col in product_details.columns:
        if product_details[col].dtype == 'object' and col not in ['product_id', 'description']:
            cat_columns.append(col)
    
    if cat_columns:
        categorical_features = pd.get_dummies(product_details[cat_columns])
    else:
        categorical_features = pd.DataFrame(index=product_details.index)
    
    # Chuẩn hóa đặc trưng số
    num_columns = []
    for col in product_details.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'product_id':  # Đảm bảo không chuẩn hóa product_id
            num_columns.append(col)
    
    if num_columns:
        numerical_features = product_details[num_columns]
        scaler = StandardScaler()
        normalized_features = pd.DataFrame(
            scaler.fit_transform(numerical_features),
            columns=numerical_features.columns,
            index=product_details.index
        )
    else:
        normalized_features = pd.DataFrame(index=product_details.index)
    
    # Nếu có dữ liệu văn bản như mô tả sản phẩm
    text_df = pd.DataFrame(index=product_details.index)
    if 'description' in product_details.columns:
        # Điền giá trị thiếu
        descriptions = product_details['description'].fillna('').tolist()
        
        if any(len(desc) > 0 for desc in descriptions):  # Kiểm tra xem có mô tả nào không
            # Xử lý văn bản với TF-IDF
            tfidf = TfidfVectorizer(max_features=100)
            text_features = tfidf.fit_transform(descriptions)
            text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
            text_df = pd.DataFrame(
                text_features.toarray(), 
                columns=text_feature_names,
                index=product_details.index
            )
    
    # Kết hợp tất cả đặc trưng
    all_features = pd.concat([categorical_features, normalized_features, text_df], axis=1)
    
    print(f"Feature extraction complete. Total features: {all_features.shape[1]}")
    
    return all_features

def train_content_based_model(product_details):
    """
    Huấn luyện mô hình Content-Based Filtering
    
    Parameters:
    -----------
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    similarity_matrix: numpy.ndarray
        Ma trận tương đồng giữa các sản phẩm
    index_to_id: dict
        Dictionary ánh xạ từ index trong ma trận tương đồng đến product_id
    id_to_index: dict
        Dictionary ánh xạ từ product_id đến index trong ma trận tương đồng
    """
    # Kiểm tra product_id có phải là cột không
    if 'product_id' not in product_details.columns:
        raise ValueError("product_details must contain 'product_id' column")
    
    # Trích xuất đặc trưng
    features = extract_features(product_details)
    
    # Tính toán ma trận tương đồng
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(features)
    
    # Lưu trữ ánh xạ giữa index và product_id cho việc tra cứu sau này
    index_to_id = dict(enumerate(product_details['product_id'].values))
    id_to_index = {id_val: idx for idx, id_val in index_to_id.items()}
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return similarity_matrix, index_to_id, id_to_index

def get_similar_products(product_id, similarity_matrix, id_to_index, index_to_id, product_details, top_n=10):
    """
    Lấy các sản phẩm tương tự dựa trên content-based filtering
    
    Parameters:
    -----------
    product_id: int
        ID của sản phẩm cần tìm các sản phẩm tương tự
    similarity_matrix: numpy.ndarray
        Ma trận tương đồng giữa các sản phẩm
    id_to_index: dict
        Dictionary ánh xạ từ product_id đến index trong ma trận tương đồng
    index_to_id: dict
        Dictionary ánh xạ từ index trong ma trận tương đồng đến product_id
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    top_n: int
        Số lượng sản phẩm tương tự cần trả về
        
    Returns:
    --------
    similar_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm tương tự
    """
    if product_id not in id_to_index:
        print(f"Product ID {product_id} not found in the database")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu không tìm thấy sản phẩm
    
    # Lấy index của sản phẩm
    idx = id_to_index[product_id]
    
    # Lấy điểm tương đồng với các sản phẩm khác
    similarity_scores = similarity_matrix[idx]
    
    # Lấy index của top N sản phẩm tương tự (bỏ qua sản phẩm hiện tại)
    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    # Lấy ID của các sản phẩm tương tự
    similar_product_ids = [index_to_id[i] for i in similar_indices]
    
    # Lấy thông tin chi tiết
    similar_products = product_details[product_details['product_id'].isin(similar_product_ids)].copy()
    
    # Thêm điểm tương đồng vào kết quả
    similarity_dict = {index_to_id[i]: similarity_scores[i] for i in similar_indices}
    similar_products['similarity_score'] = similar_products['product_id'].map(similarity_dict)
    
    # Sắp xếp theo điểm tương đồng giảm dần
    similar_products = similar_products.sort_values(by='similarity_score', ascending=False)
    
    return similar_products