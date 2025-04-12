import pandas as pd

def preprocess_events(events):
    # Nếu `events` là một DataFrame, sử dụng trực tiếp
    if isinstance(events, pd.DataFrame):
        return events.groupby(['user_id', 'product_id', 'event_type']).size().unstack(fill_value=0)
    
    # Nếu `events` là một đường dẫn, đọc file CSV
    events = pd.read_csv(events)
    return events.groupby(['user_id', 'product_id', 'event_type']).size().unstack(fill_value=0)


def calculate_relevance_score(product_id, top_products, product_details):
    # Tính điểm liên quan dựa trên các đặc trưng (ví dụ: product_type, giá, đánh giá, v.v.)
    relevance_score = 0
    
    # Lấy thông tin sản phẩm hiện tại
    current_product = product_details[product_details['product_id'] == product_id]
    
    for top_product_id in top_products:
        top_product = product_details[product_details['product_id'] == top_product_id]
        
        # Tăng điểm nếu `product_type` giống nhau
        if current_product['product_type'].values[0] == top_product['product_type'].values[0]:
            relevance_score += 1
        
        # Tăng điểm nếu giá gần nhau (ví dụ: chênh lệch dưới 20%)
        if abs(current_product['price'].values[0] - top_product['price'].values[0]) / top_product['price'].values[0] < 0.2:
            relevance_score += 0.5
        
        # Tăng điểm nếu đánh giá gần nhau (ví dụ: chênh lệch dưới 1 sao)
        if abs(current_product['rating'].values[0] - top_product['rating'].values[0]) < 1:
            relevance_score += 0.5
    
    return relevance_score

def recommend_based_on_events(user_id, events_path, product_details, top_n=10):
    # Tiền xử lý dữ liệu sự kiện
    user_product_events = preprocess_events(events_path)
    
    if user_id not in user_product_events.index:
        return pd.DataFrame()  # Trả về rỗng nếu không có sự kiện cho người dùng
    
    # Lấy danh sách sản phẩm mà người dùng đã tương tác
    user_behavior = user_product_events.loc[user_id]
    
    # Tính tổng điểm cho từng sản phẩm
    user_behavior['score'] = user_behavior.sum(axis=1)
    
    # Lấy danh sách các sản phẩm thao tác nhiều nhất (top 3)
    top_products = user_behavior.sort_values(by='score', ascending=False).head(3).index.tolist()
    
    # Lấy danh sách các `product_type` tương ứng
    top_product_types = product_details.loc[
        product_details['product_id'].isin(top_products), 'product_type'
    ].unique()
    
    # Gợi ý các sản phẩm có `product_type` tương tự
    recommended_products = product_details[
        product_details['product_type'].isin(top_product_types)
    ]
    
    # Sắp xếp sản phẩm gợi ý theo mức độ liên quan (nếu cần)
    recommended_products['relevance_score'] = recommended_products['product_id'].apply(
        lambda pid: calculate_relevance_score(pid, top_products, product_details)
    )
    recommended_products = recommended_products.sort_values(by='relevance_score', ascending=False).head(top_n)
    
    return recommended_products