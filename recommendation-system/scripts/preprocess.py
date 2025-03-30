import pandas as pd

def preprocess_data(ratings_path, products_path, events_path):
    ratings = pd.read_csv(ratings_path)
    products = pd.read_csv(products_path)
    events = pd.read_csv(events_path)
    # Tích hợp dữ liệu
    integrated_data = pd.merge(ratings, products, on='product_id', how='left')
    return integrated_data