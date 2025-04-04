from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def train_content_based_model(product_details):
    # Tạo đặc trưng từ thông tin sản phẩm
    product_features = pd.get_dummies(product_details[['product_type', 'location', 'price', 'rating']])
    
    # Tính toán ma trận tương đồng
    similarity_matrix = cosine_similarity(product_features)
    
    return similarity_matrix