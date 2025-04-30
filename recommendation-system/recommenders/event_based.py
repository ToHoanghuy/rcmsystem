import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_events(events):
    # Nếu `events` là một DataFrame, sử dụng trực tiếp
    if isinstance(events, pd.DataFrame):
        return events.groupby(['user_id', 'product_id', 'event_type']).size().unstack(fill_value=0)
    
    # Nếu `events` là một đường dẫn, đọc file CSV
    events = pd.read_csv(events)
    return events.groupby(['user_id', 'product_id', 'event_type']).size().unstack(fill_value=0)

def preprocess_events_advanced(events_path):
    """
    Tiền xử lý dữ liệu sự kiện nâng cao với tính thời gian
    
    Parameters:
    -----------
    events_path: str
        Đường dẫn đến file dữ liệu sự kiện
        
    Returns:
    --------
    user_product_scores: DataFrame
        DataFrame chứa điểm số sự kiện cho từng cặp người dùng và sản phẩm
    """
    # Đọc dữ liệu sự kiện
    events = pd.read_csv(events_path)
    
    # Nếu có cột timestamp, chuyển đổi thành datetime
    if 'timestamp' in events.columns:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        
        # Tính thời gian đã trôi qua (theo ngày)
        current_time = datetime.now()
        events['days_ago'] = (current_time - events['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Tính hệ số thời gian (gần đây hơn sẽ có trọng số cao hơn)
        events['time_factor'] = np.exp(-0.1 * events['days_ago'])  # Hàm decay theo thời gian
    else:
        # Nếu không có timestamp, gán hệ số thời gian bằng 1
        events['time_factor'] = 1.0
    
    # Gán trọng số cho từng loại sự kiện
    event_weights = {
        'detail_product_view': 1.0,
        'add_to_cart': 2.5,
        'purchase': 5.0,
        'home_page_view': 0.5,
        'search': 0.8,
        'view': 1.0,
        'like': 2.0,
        'share': 3.0,
        'comment': 2.5,
        'bookmark': 4.0
    }
    
    # Mặc định cho các sự kiện không được định nghĩa
    default_weight = 1.0
    
    # Tính điểm sự kiện có trọng số và tính thời gian
    events['event_score'] = events.apply(
        lambda row: row['time_factor'] * event_weights.get(row['event_type'], default_weight),
        axis=1
    )
    
    # Tổng hợp điểm theo user_id và product_id
    user_product_scores = events.groupby(['user_id', 'product_id'])['event_score'].sum().reset_index()
    
    return user_product_scores

def analyze_user_behavior_sequence(events_path):
    """
    Phân tích chuỗi hành vi của người dùng
    
    Parameters:
    -----------
    events_path: str
        Đường dẫn đến file dữ liệu sự kiện
        
    Returns:
    --------
    transition_probs: dict
        Dictionary chứa xác suất chuyển đổi giữa các loại sự kiện
    """
    # Đọc dữ liệu sự kiện
    events = pd.read_csv(events_path)
    
    # Nếu có timestamp, sắp xếp theo thời gian
    if 'timestamp' in events.columns:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        events = events.sort_values(by=['user_id', 'timestamp'])
    
    # Tính xác suất chuyển đổi giữa các loại sự kiện
    transition_counts = {}
    
    for user_id in events['user_id'].unique():
        user_events = events[events['user_id'] == user_id]
        if len(user_events) < 2:
            continue
        
        for i in range(len(user_events) - 1):
            current_event = user_events.iloc[i]['event_type']
            next_event = user_events.iloc[i + 1]['event_type']
            current_product = user_events.iloc[i]['product_id']
            next_product = user_events.iloc[i + 1]['product_id']
            
            # Lưu thông tin chuyển đổi
            key = (current_event, current_product, next_event)
            if key in transition_counts:
                transition_counts[key] += 1
            else:
                transition_counts[key] = 1
    
    # Tính xác suất chuyển đổi
    transition_probs = {}
    for (current_event, current_product, next_event), count in transition_counts.items():
        total = sum(v for (e, p, _), v in transition_counts.items() if e == current_event and p == current_product)
        transition_probs[(current_event, current_product, next_event)] = count / total
    
    return transition_probs

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

def recommend_based_on_events_advanced(user_id, events_path, product_details, transition_probs=None, top_n=10):
    """
    Gợi ý nâng cao dựa trên sự kiện người dùng
    
    Parameters:
    -----------
    user_id: int
        ID của người dùng cần gợi ý
    events_path: str
        Đường dẫn đến file dữ liệu sự kiện
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    transition_probs: dict, optional
        Dictionary chứa xác suất chuyển đổi giữa các loại sự kiện
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
        
    Returns:
    --------
    recommended_products: DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    """
    # Tiền xử lý dữ liệu sự kiện
    user_product_scores = preprocess_events_advanced(events_path)
    
    # Lọc theo user_id
    user_scores = user_product_scores[user_product_scores['user_id'] == user_id]
    
    if user_scores.empty:
        # Nếu không có dữ liệu sự kiện, trả về sản phẩm phổ biến
        popular_products = product_details.sort_values(by='rating', ascending=False).head(top_n)
        return popular_products
    
    # Tìm sự kiện gần đây nhất của người dùng
    events = pd.read_csv(events_path)
    if 'timestamp' in events.columns:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        latest_events = events[(events['user_id'] == user_id)].sort_values(by='timestamp', ascending=False).head(3)
        
        if not latest_events.empty and transition_probs:
            # Nếu có dữ liệu chuyển đổi, dự đoán sản phẩm tiếp theo
            next_product_scores = {}
            
            for _, event in latest_events.iterrows():
                current_event = event['event_type']
                current_product = event['product_id']
                
                # Tìm các sản phẩm tiếp theo có khả năng cao
                for (e, p, next_e), prob in transition_probs.items():
                    if e == current_event and p == current_product:
                        for _, next_product_row in product_details.iterrows():
                            next_product_id = next_product_row['product_id']
                            if next_product_id != current_product:  # Không gợi ý sản phẩm hiện tại
                                if next_product_id in next_product_scores:
                                    next_product_scores[next_product_id] += prob
                                else:
                                    next_product_scores[next_product_id] = prob
                                
            # Kết hợp với điểm sự kiện
            for _, row in user_scores.iterrows():
                product_id = row['product_id']
                event_score = row['event_score']
                if product_id in next_product_scores:
                    next_product_scores[product_id] += event_score * 0.5
                else:
                    next_product_scores[product_id] = event_score * 0.5
        else:
            # Nếu không có dữ liệu chuyển đổi, sử dụng điểm sự kiện đơn thuần
            next_product_scores = {row['product_id']: row['event_score'] for _, row in user_scores.iterrows()}
    else:
        # Nếu không có timestamp, sử dụng điểm sự kiện đơn thuần
        next_product_scores = {row['product_id']: row['event_score'] for _, row in user_scores.iterrows()}
    
    # Lấy top N sản phẩm
    top_products = sorted(next_product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_product_ids = [product_id for product_id, _ in top_products]
    
    # Lấy thông tin chi tiết
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    if not recommended_products.empty:
        score_dict = {pid: score for pid, score in top_products}
        recommended_products['event_score'] = recommended_products['product_id'].map(score_dict)
        recommended_products = recommended_products.sort_values(by='event_score', ascending=False)
    
    return recommended_products