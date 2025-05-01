import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_events(events):
    """
    Tiền xử lý dữ liệu sự kiện cơ bản
    
    Parameters:
    -----------
    events : DataFrame or str
        DataFrame chứa dữ liệu sự kiện hoặc đường dẫn đến file events
        
    Returns:
    --------
    DataFrame
        DataFrame đã được pivot theo user_id, product_id và event_type
    """
    logger.info("Preprocessing events data")
    
    # Kiểm tra đầu vào
    if isinstance(events, str):
        logger.info(f"Loading events from file: {events}")
        events = pd.read_csv(events)
    
    # Kiểm tra cấu trúc dữ liệu
    if not isinstance(events, pd.DataFrame) or events.empty:
        logger.warning("Empty events data provided")
        return pd.DataFrame()
    
    required_cols = ['user_id', 'event_type', 'product_id']
    if not all(col in events.columns for col in required_cols):
        logger.warning(f"Missing required columns in events data. Required: {required_cols}")
        return pd.DataFrame()
    
    # Chuyển đổi sang pivot table
    try:
        user_product_events = events.groupby(['user_id', 'product_id', 'event_type']).size().unstack(fill_value=0)
        logger.info(f"Created event pivot table with shape {user_product_events.shape}")
        return user_product_events
    except Exception as e:
        logger.error(f"Error creating pivot table: {str(e)}")
        return pd.DataFrame()

def preprocess_events_advanced(events_path):
    """
    Tiền xử lý dữ liệu sự kiện nâng cao với tính thời gian
    
    Parameters:
    -----------
    events_path: str or DataFrame
        Đường dẫn đến file dữ liệu sự kiện hoặc DataFrame chứa dữ liệu sự kiện
        
    Returns:
    --------
    user_product_scores: DataFrame
        DataFrame chứa điểm số sự kiện cho từng cặp người dùng và sản phẩm
    """
    logger.info("Running advanced event preprocessing")
    
    # Xử lý đầu vào
    if isinstance(events_path, pd.DataFrame):
        events = events_path
    else:
        try:
            events = pd.read_csv(events_path)
        except Exception as e:
            logger.error(f"Failed to read events data: {str(e)}")
            return pd.DataFrame()
    
    # Nếu có cột timestamp, chuyển đổi thành datetime
    if 'timestamp' in events.columns:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        
        # Tính thời gian đã trôi qua (theo ngày)
        current_time = datetime.now()
        events['days_ago'] = (current_time - events['timestamp']).dt.total_seconds() / (24 * 3600)
        
        # Tính hệ số thời gian (gần đây hơn sẽ có trọng số cao hơn)
        events['time_factor'] = np.exp(-0.05 * events['days_ago'])  # Điều chỉnh hệ số 0.05 để giảm tốc độ decay
    else:
        # Nếu không có timestamp, gán hệ số thời gian bằng 1
        events['time_factor'] = 1.0
    
    # Đảm bảo user_id và product_id có kiểu int
    events['user_id'] = events['user_id'].astype(int)
    events['product_id'] = events['product_id'].astype(int)
    
    # Gán trọng số cho từng loại sự kiện - điều chỉnh trọng số dựa trên tầm quan trọng
    event_weights = {
        'detail_product_view': 1.5,     # Tăng từ 1.0 -> 1.5
        'add_to_cart': 3.0,             # Tăng từ 2.5 -> 3.0  
        'purchase': 5.0,                # Giữ nguyên - đây là hành động quan trọng nhất
        'home_page_view': 0.5,          # Giữ nguyên - ít quan trọng nhất
        'search': 1.0,                  # Tăng từ 0.8 -> 1.0
        'view': 1.0,                    # Giữ nguyên
        'like': 2.0,                    # Giữ nguyên
        'share': 3.0,                   # Giữ nguyên
        'comment': 2.5,                 # Giữ nguyên
        'bookmark': 4.0                 # Giữ nguyên
    }
    
    # Mặc định cho các sự kiện không được định nghĩa
    default_weight = 0.5  # Giảm từ 1.0 -> 0.5 cho các sự kiện không xác định
    
    # Tính điểm sự kiện có trọng số và tính thời gian
    events['event_score'] = events.apply(
        lambda row: row['time_factor'] * event_weights.get(row['event_type'], default_weight),
        axis=1
    )
    
    # Tổng hợp điểm theo user_id và product_id
    user_product_scores = events.groupby(['user_id', 'product_id']).agg(
        event_score=pd.NamedAgg(column='event_score', aggfunc='sum'),
        event_count=pd.NamedAgg(column='event_type', aggfunc='count'),
        last_interaction=pd.NamedAgg(column='timestamp', aggfunc='max') if 'timestamp' in events.columns else None
    ).reset_index()
    
    # Chuẩn hóa điểm về thang 1-5 để phù hợp với thang đánh giá
    if not user_product_scores.empty and user_product_scores['event_score'].max() > 0:
        max_score = user_product_scores['event_score'].max()
        user_product_scores['normalized_score'] = (user_product_scores['event_score'] / max_score * 4) + 1
    else:
        user_product_scores['normalized_score'] = 1.0
    
    logger.info(f"Generated event scores for {len(user_product_scores)} user-product pairs")
    return user_product_scores

def analyze_user_behavior_sequence(events):
    """
    Phân tích chuỗi hành vi của người dùng
    
    Parameters:
    -----------
    events: str or DataFrame
        Đường dẫn đến file dữ liệu sự kiện hoặc DataFrame chứa dữ liệu sự kiện
        
    Returns:
    --------
    transition_probs: dict
        Dictionary chứa xác suất chuyển đổi giữa các loại sự kiện
    """
    logger.info("Analyzing user behavior sequences")
    
    # Đọc dữ liệu sự kiện
    if isinstance(events, str):
        try:
            events = pd.read_csv(events)
        except Exception as e:
            logger.error(f"Failed to read events data for behavior analysis: {str(e)}")
            return {}
    
    # Nếu có timestamp, sắp xếp theo thời gian
    if 'timestamp' in events.columns:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        events = events.sort_values(by=['user_id', 'timestamp'])
    else:
        logger.warning("No timestamp column found for sequence analysis. Results may be less accurate.")
    
    # Tính xác suất chuyển đổi giữa các loại sự kiện
    transition_counts = {}
    user_count = 0
    
    for user_id in events['user_id'].unique():
        user_events = events[events['user_id'] == user_id]
        if len(user_events) < 2:
            continue
        
        user_count += 1
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
    
    logger.info(f"Analyzed event sequences for {user_count} users")
    
    # Tính xác suất chuyển đổi
    transition_probs = {}
    for (current_event, current_product, next_event), count in transition_counts.items():
        total = sum(v for (e, p, _), v in transition_counts.items() if e == current_event and p == current_product)
        transition_probs[(current_event, current_product, next_event)] = count / total
    
    logger.info(f"Generated {len(transition_probs)} transition probabilities")
    return transition_probs

def calculate_relevance_score(product_id, top_products, product_details):
    """
    Tính điểm tương đồng giữa một sản phẩm với danh sách sản phẩm top
    
    Parameters:
    -----------
    product_id: int
        ID của sản phẩm cần tính điểm
    top_products: list
        Danh sách ID của các sản phẩm top
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
        
    Returns:
    --------
    float
        Điểm tương đồng
    """
    # Tính điểm liên quan dựa trên các đặc trưng
    relevance_score = 0
    
    # Lấy thông tin sản phẩm hiện tại
    current_product = product_details[product_details['product_id'] == product_id]
    if current_product.empty:
        return 0
    
    for top_product_id in top_products:
        top_product = product_details[product_details['product_id'] == top_product_id]
        if top_product.empty:
            continue
        
        # Tăng điểm nếu `product_type` giống nhau - đây là yếu tố quan trọng nhất
        if 'product_type' in current_product.columns and 'product_type' in top_product.columns:
            try:
                if current_product['product_type'].iloc[0] == top_product['product_type'].iloc[0]:
                    relevance_score += 2.0  # Tăng từ 1.0 -> 2.0
            except:
                pass
        
        # Tăng điểm nếu giá gần nhau (ví dụ: chênh lệch dưới 20%)
        if 'price' in current_product.columns and 'price' in top_product.columns:
            try:
                top_price = float(top_product['price'].iloc[0])
                if top_price > 0:
                    cur_price = float(current_product['price'].iloc[0])
                    if abs(cur_price - top_price) / top_price < 0.2:
                        relevance_score += 1.0  # Tăng từ 0.5 -> 1.0
            except:
                pass
        
        # Tăng điểm nếu đánh giá gần nhau (ví dụ: chênh lệch dưới 1 sao)
        if 'rating' in current_product.columns and 'rating' in top_product.columns:
            try:
                if abs(float(current_product['rating'].iloc[0]) - float(top_product['rating'].iloc[0])) < 0.5:  # Giảm từ 1.0 -> 0.5
                    relevance_score += 0.8  # Tăng từ 0.5 -> 0.8
            except:
                pass
    
    return relevance_score

def recommend_based_on_events(user_id, events_path, product_details, top_n=10):
    """
    Gợi ý dựa trên sự kiện người dùng
    
    Parameters:
    -----------
    user_id: int
        ID của người dùng cần gợi ý
    events_path: str or DataFrame
        Đường dẫn đến file dữ liệu sự kiện hoặc DataFrame chứa dữ liệu sự kiện
    product_details: DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm
    top_n: int
        Số lượng sản phẩm gợi ý cần trả về
        
    Returns:
    --------
    DataFrame
        DataFrame chứa thông tin về các sản phẩm được gợi ý
    """
    logger.info(f"Generating recommendations for user {user_id} based on events")
    
    # Kiểm tra product_details
    if product_details is None or not isinstance(product_details, pd.DataFrame) or product_details.empty:
        logger.warning("No product details provided")
        return pd.DataFrame()
    
    # Tiền xử lý dữ liệu sự kiện
    user_product_events = preprocess_events(events_path)
    
    if user_product_events.empty:
        logger.warning("No event data available after preprocessing")
        # Trả về sản phẩm phổ biến nếu không có dữ liệu sự kiện
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Kiểm tra nếu user_id không có trong dữ liệu
    if user_id not in user_product_events.index:
        logger.warning(f"User {user_id} not found in event data")
        # Trả về sản phẩm phổ biến nếu người dùng không có sự kiện
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Lấy danh sách sản phẩm mà người dùng đã tương tác
    try:
        user_behavior = user_product_events.loc[user_id]
        
        # Tính tổng điểm cho từng sản phẩm
        if isinstance(user_behavior, pd.Series):
            # Nếu user chỉ có 1 sản phẩm, convert Series thành DataFrame
            user_behavior = pd.DataFrame([user_behavior])
            user_behavior.index = [user_behavior.index.values[0]]
            user_behavior['score'] = user_behavior.sum(axis=1)
        else:
            user_behavior['score'] = user_behavior.sum(axis=1)
        
        # Lấy danh sách các sản phẩm thao tác nhiều nhất (top 3)
        top_products = user_behavior.sort_values(by='score', ascending=False).head(3).index.tolist()
    except Exception as e:
        logger.error(f"Error processing user behavior: {str(e)}")
        top_products = []
    
    if not top_products:
        logger.warning("No top products found for user")
        # Trả về sản phẩm phổ biến nếu không tìm thấy sản phẩm top
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Lấy danh sách các `product_type` tương ứng
    try:
        top_product_types = product_details.loc[
            product_details['product_id'].isin(top_products), 'product_type'
        ].unique()
    except Exception as e:
        logger.error(f"Error finding product types: {str(e)}")
        top_product_types = []
    
    # Nếu không tìm thấy product_type, trả về sản phẩm phổ biến
    if not len(top_product_types):
        logger.warning("No product types found for top products")
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Gợi ý các sản phẩm có `product_type` tương tự
    recommended_products = product_details[
        product_details['product_type'].isin(top_product_types)
    ].copy()
    
    # Loại bỏ các sản phẩm mà người dùng đã tương tác
    recommended_products = recommended_products[~recommended_products['product_id'].isin(top_products)]
    
    if recommended_products.empty:
        logger.warning("No recommended products found after filtering")
        if 'rating' in product_details.columns:
            return product_details.sort_values(by='rating', ascending=False).head(top_n)
        else:
            return product_details.head(top_n)
    
    # Sắp xếp sản phẩm gợi ý theo mức độ liên quan
    try:
        recommended_products['relevance_score'] = recommended_products['product_id'].apply(
            lambda pid: calculate_relevance_score(pid, top_products, product_details)
        )
        recommended_products = recommended_products.sort_values(by='relevance_score', ascending=False).head(top_n)
    except Exception as e:
        logger.error(f"Error calculating relevance scores: {str(e)}")
        # Nếu tính toán relevance_score bị lỗi, sử dụng rating để sắp xếp
        if 'rating' in recommended_products.columns:
            recommended_products = recommended_products.sort_values(by='rating', ascending=False).head(top_n)
        else:
            recommended_products = recommended_products.head(top_n)
    
    logger.info(f"Generated {len(recommended_products)} recommendations based on events")
    return recommended_products

def recommend_based_on_events_advanced(user_id, events_path, product_details, transition_probs=None, top_n=10):
    """
    Gợi ý nâng cao dựa trên sự kiện người dùng
    
    Parameters:
    -----------
    user_id: int
        ID của người dùng cần gợi ý
    events_path: str or DataFrame
        Đường dẫn đến file dữ liệu sự kiện hoặc DataFrame chứa dữ liệu sự kiện
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
    logger.info(f"Generating advanced recommendations for user {user_id}")
    
    # Tiền xử lý dữ liệu sự kiện
    user_product_scores = preprocess_events_advanced(events_path)
    
    if user_product_scores.empty:
        logger.warning("No event data available after advanced preprocessing")
        # Trả về sản phẩm phổ biến nếu không có dữ liệu sự kiện
        popular_products = product_details.sort_values(by='rating', ascending=False).head(top_n) if 'rating' in product_details.columns else product_details.head(top_n)
        return popular_products
    
    # Lọc theo user_id
    user_scores = user_product_scores[user_product_scores['user_id'] == user_id]
    
    if user_scores.empty:
        logger.warning(f"User {user_id} not found in event data")
        # Nếu không có dữ liệu sự kiện, trả về sản phẩm phổ biến
        popular_products = product_details.sort_values(by='rating', ascending=False).head(top_n) if 'rating' in product_details.columns else product_details.head(top_n)
        return popular_products
    
    # Tìm sự kiện gần đây nhất của người dùng
    events = pd.read_csv(events_path) if isinstance(events_path, str) else events_path.copy()
    
    next_product_scores = {}
    
    # Nếu có timestamp và transition_probs, sử dụng phân tích chuỗi hành vi
    if 'timestamp' in events.columns and transition_probs:
        events['timestamp'] = pd.to_datetime(events['timestamp'])
        latest_events = events[(events['user_id'] == user_id)].sort_values(by='timestamp', ascending=False).head(3)
        
        if not latest_events.empty:
            # Dự đoán sản phẩm tiếp theo dựa trên chuỗi hành vi
            for _, event in latest_events.iterrows():
                current_event = event['event_type']
                current_product = event['product_id']
                
                # Tìm các sản phẩm tiếp theo có khả năng cao
                for (e, p, next_e), prob in transition_probs.items():
                    if e == current_event and p == current_product:
                        # Chỉ xem xét các sản phẩm hiện có trong product_details
                        for _, next_product_row in product_details.iterrows():
                            next_product_id = next_product_row['product_id']
                            if next_product_id != current_product:  # Không gợi ý sản phẩm hiện tại
                                weight = prob * 2  # Tăng mức độ ảnh hưởng của transition probability
                                if next_product_id in next_product_scores:
                                    next_product_scores[next_product_id] += weight
                                else:
                                    next_product_scores[next_product_id] = weight
    
    # Kết hợp với event score
    for _, row in user_scores.iterrows():
        product_id = row['product_id']
        event_score = row['event_score'] if 'event_score' in row else (row['normalized_score'] if 'normalized_score' in row else 1.0)
        
        # Loại bỏ các sản phẩm người dùng đã tương tác nhiều lần (giả định họ đã mua hoặc không quan tâm)
        if row['event_count'] > 10:  # Ngưỡng số lượng tương tác tối đa
            continue
            
        if product_id in next_product_scores:
            next_product_scores[product_id] += event_score * 0.8  # Giảm trọng số của điểm sự kiện
        else:
            next_product_scores[product_id] = event_score * 0.8
    
    # Nếu không có sản phẩm được gợi ý, trả về sản phẩm phổ biến
    if not next_product_scores:
        logger.warning("No product scores generated, falling back to popular products")
        return product_details.sort_values(by='rating', ascending=False).head(top_n) if 'rating' in product_details.columns else product_details.head(top_n)
    
    # Lấy top N sản phẩm
    top_products = sorted(next_product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_product_ids = [product_id for product_id, _ in top_products]
    
    # Lấy thông tin chi tiết
    recommended_products = product_details[product_details['product_id'].isin(top_product_ids)].copy()
    
    if recommended_products.empty:
        logger.warning("No recommended products found in product details")
        return product_details.sort_values(by='rating', ascending=False).head(top_n) if 'rating' in product_details.columns else product_details.head(top_n)
    
    # Thêm điểm sự kiện vào kết quả
    score_dict = {pid: score for pid, score in top_products}
    recommended_products['event_score'] = recommended_products['product_id'].map(score_dict)
    recommended_products = recommended_products.sort_values(by='event_score', ascending=False)
    
    logger.info(f"Generated {len(recommended_products)} advanced recommendations")
    return recommended_products

def process_feedback_loop(events_path, recommendations, feedback_data=None):
    """
    Process feedback loop to improve recommendations over time
    
    Parameters:
    -----------
    events_path: str or DataFrame
        Path to the events data file or DataFrame containing event data
    recommendations: DataFrame
        DataFrame containing recommended products and their scores
    feedback_data: DataFrame, optional
        DataFrame containing explicit feedback from users on recommendations
        
    Returns:
    --------
    DataFrame
        Updated recommendations with adjusted scores based on feedback
    """
    logger.info("Processing feedback loop to improve recommendations")
    
    if recommendations.empty:
        logger.warning("No recommendations to process in feedback loop")
        return recommendations
    
    # If we have explicit feedback data, process it
    if feedback_data is not None and not feedback_data.empty:
        try:
            # Merge feedback with recommendations
            merged_data = recommendations.merge(
                feedback_data,
                on=['user_id', 'product_id'],
                how='left'
            )
            
            # Adjust scores based on explicit feedback
            merged_data['adjusted_score'] = merged_data.apply(
                lambda row: row.get('normalized_score', 0) * (1 + 0.2 * row.get('feedback_score', 0)),
                axis=1
            )
            
            logger.info(f"Applied explicit feedback adjustments to {len(merged_data)} recommendations")
            return merged_data.sort_values(by='adjusted_score', ascending=False)
        except Exception as e:
            logger.error(f"Error processing explicit feedback: {str(e)}")
            # If there's an error, continue with implicit feedback
    
    # Process implicit feedback from events
    try:
        # Read events data if it's a file path
        if isinstance(events_path, str):
            events = pd.read_csv(events_path)
        else:
            events = events_path
            
        # Get recent events (last 30 days)
        if 'timestamp' in events.columns:
            events['timestamp'] = pd.to_datetime(events['timestamp'])
            current_time = datetime.now()
            recent_events = events[events['timestamp'] > (current_time - pd.Timedelta(days=30))]
        else:
            recent_events = events
            
        # Calculate engagement metrics for recommended products
        product_engagement = recent_events.groupby('product_id').agg(
            view_count=pd.NamedAgg(column='event_type', aggfunc=lambda x: (x == 'detail_product_view').sum()),
            cart_count=pd.NamedAgg(column='event_type', aggfunc=lambda x: (x == 'add_to_cart').sum()),
            purchase_count=pd.NamedAgg(column='event_type', aggfunc=lambda x: (x == 'purchase').sum())
        )
        
        # Normalize engagement metrics
        if not product_engagement.empty:
            for col in ['view_count', 'cart_count', 'purchase_count']:
                max_val = product_engagement[col].max()
                if max_val > 0:
                    product_engagement[f'norm_{col}'] = product_engagement[col] / max_val
                else:
                    product_engagement[f'norm_{col}'] = 0
                    
            # Calculate engagement score (weighted sum of normalized metrics)
            product_engagement['engagement_score'] = (
                0.2 * product_engagement['norm_view_count'] + 
                0.3 * product_engagement['norm_cart_count'] + 
                0.5 * product_engagement['norm_purchase_count']
            )
            
            # Merge engagement scores with recommendations
            recommendations = recommendations.merge(
                product_engagement[['engagement_score']], 
                left_on='product_id', 
                right_index=True,
                how='left'
            )
            
            # Fill missing engagement scores with 0
            recommendations['engagement_score'] = recommendations['engagement_score'].fillna(0)
            
            # Calculate adjusted scores
            if 'normalized_score' in recommendations.columns:
                recommendations['adjusted_score'] = (
                    0.7 * recommendations['normalized_score'] + 
                    0.3 * recommendations['engagement_score']
                )
            else:
                recommendations['adjusted_score'] = recommendations['engagement_score']
                
            logger.info(f"Applied implicit feedback adjustments to {len(recommendations)} recommendations")
            return recommendations.sort_values(by='adjusted_score', ascending=False)
        else:
            logger.warning("No engagement data available for feedback loop")
            return recommendations
            
    except Exception as e:
        logger.error(f"Error processing implicit feedback: {str(e)}")
        return recommendations

def recommend_with_feedback(user_id, events_path, product_details, feedback_data=None, top_n=10):
    """
    Generate recommendations with feedback loop integration
    
    Parameters:
    -----------
    user_id: int
        ID of the user to generate recommendations for
    events_path: str or DataFrame
        Path to the events data file or DataFrame containing event data
    product_details: DataFrame
        DataFrame containing product details
    feedback_data: DataFrame, optional
        DataFrame containing explicit feedback from users on recommendations
    top_n: int
        Number of recommendations to return
        
    Returns:
    --------
    DataFrame
        DataFrame containing recommended products with feedback-adjusted scores
    """
    logger.info(f"Generating recommendations with feedback loop for user {user_id}")
    
    # First get base recommendations
    base_recommendations = recommend_based_on_events_advanced(
        user_id, events_path, product_details, top_n=top_n*2
    )
    
    if base_recommendations.empty:
        logger.warning("No base recommendations generated")
        return pd.DataFrame()
    
    # Add user_id column for feedback processing
    base_recommendations['user_id'] = user_id
    
    # Process feedback loop
    adjusted_recommendations = process_feedback_loop(
        events_path, base_recommendations, feedback_data
    )
    
    # Return top N recommendations
    final_recommendations = adjusted_recommendations.head(top_n)
    logger.info(f"Generated {len(final_recommendations)} recommendations with feedback adjustment")
    
    return final_recommendations