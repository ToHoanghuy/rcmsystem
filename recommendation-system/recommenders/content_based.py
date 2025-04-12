from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def train_content_based_model(product_details):
    # Tính toán tương đồng cho từng đặc trưng
    product_type_similarity = cosine_similarity(pd.get_dummies(product_details['product_type']))
    location_similarity = cosine_similarity(pd.get_dummies(product_details['location']))
    
    scaler = MinMaxScaler()
    numerical_features = scaler.fit_transform(product_details[['price', 'rating']])
    numerical_similarity = cosine_similarity(numerical_features)
    
    # Gán trọng số cho từng đặc trưng
    similarity_matrix = (
        0.4 * product_type_similarity +
        0.3 * location_similarity +
        0.3 * numerical_similarity
    )
    
    return similarity_matrix