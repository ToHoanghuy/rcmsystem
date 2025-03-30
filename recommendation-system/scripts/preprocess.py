import pandas as pd

def preprocess_data(ratings_path, products_path, events_path):
    # Đọc dữ liệu đánh giá
    ratings = pd.read_csv(ratings_path)
    # Đọc dữ liệu sản phẩm
    products = pd.read_csv(products_path)
    # Đọc dữ liệu sự kiện (nếu cần)
    events = pd.read_csv(events_path)
    
    # Tích hợp dữ liệu
    integrated_data = pd.merge(ratings, products, on='product_id', how='left')
    
    # Đổi tên cột rating_x thành rating
    if 'rating_x' in integrated_data.columns:
        integrated_data.rename(columns={'rating_x': 'rating'}, inplace=True)
    
    return integrated_data