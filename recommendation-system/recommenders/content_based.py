from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def train_content_based_model(product_details):
    product_features = pd.get_dummies(product_details[['product_type', 'location', 'price', 'rating']])
    similarity_matrix = cosine_similarity(product_features)
    return similarity_matrix