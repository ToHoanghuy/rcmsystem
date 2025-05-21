import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_distribution(data):
    """
    Phân tích phân phối dữ liệu và trả về báo cáo thống kê
    """
    # Tính toán các chỉ số thống kê cơ bản
    stats = {
        'shape': data.shape,
        'missing_values': data.isnull().sum(),
        'data_types': data.dtypes,
        'numeric_summary': data.describe(),
    }
    
    # Thêm phân tích cho cột categorical nếu có
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        stats['categorical_summary'] = {col: data[col].value_counts() for col in cat_cols}
        
    return stats

def load_data(filepath):
    """
    Load dữ liệu từ file và trả về DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File không tồn tại: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        return pd.read_csv(filepath)
    elif ext == '.json':
        return pd.read_json(filepath)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")

def clean_data(data):
    """
    Làm sạch dữ liệu bằng cách xử lý missing values và outliers
    """
    # Loại bỏ các dòng thiếu user_id, product_id, rating (nếu có)
    required_cols = [col for col in ['user_id', 'product_id', 'rating'] if col in data.columns]
    if required_cols:
        before = len(data)
        data = data.dropna(subset=required_cols)
        after = len(data)
        if after < before:
            logger.warning(f"Đã loại bỏ {before - after} dòng thiếu thông tin quan trọng: {required_cols}")

    # Chuẩn hóa rating về scale 1-5 nếu có rating
    if 'rating' in data.columns:
        min_rating = data['rating'].min()
        max_rating = data['rating'].max()
        if min_rating < 1 or max_rating > 5:
            logger.info(f"Chuẩn hóa rating về scale 1-5 (min={min_rating}, max={max_rating})")
            data['rating'] = 1 + 4 * (data['rating'] - min_rating) / (max_rating - min_rating)
            data['rating'] = data['rating'].clip(1, 5)

    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values based on column type
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
        else:
            data[col] = data[col].fillna(data[col].median() if not data[col].empty else 0)
    
    # Handle outliers using IQR method for numeric columns
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    return data

def normalize_data(data):
    """
    Chuẩn hóa dữ liệu sử dụng StandardScaler cho các cột số phù hợp
    """
    # Xác định các cột cần chuẩn hóa - loại trừ các ID, rating và timestamp
    exclude_cols = ['user_id', 'product_id', 'rating', 'timestamp', 'event_weight']
    numeric_cols = [col for col in data.select_dtypes(include=['float64', 'int64']).columns 
                   if col not in exclude_cols]
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        normalized_numeric = pd.DataFrame(
            scaler.fit_transform(data[numeric_cols]),
            columns=numeric_cols
        )
        
        # Replace columns in original data
        for col in numeric_cols:
            data[col] = normalized_numeric[col]
    
    return data

def preprocess(filepath):
    """
    Function tổng hợp để xử lý dữ liệu từ file
    """
    data = load_data(filepath)
    # Analyze data distribution
    analyze_data_distribution(data)
    # Clean data
    cleaned_data = clean_data(data)
    # Normalize data
    normalized_data = normalize_data(cleaned_data)
    return normalized_data

def load_product_details(filepath):
    """
    Load thông tin chi tiết về sản phẩm
    """
    products = load_data(filepath)
    
    # Check for NaN values and report
    null_counts = products.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found {null_counts.sum()} NaN values in product details data")
        logger.warning(f"NaN counts per column:\n{null_counts[null_counts > 0]}")
    
    # Check if 'name' column exists and rename it to 'product_name' if needed
    if 'name' in products.columns and 'product_name' not in products.columns:
        products['product_name'] = products['name']
    
    # Đảm bảo có các cột cần thiết
    required_columns = ['product_id']
    missing_columns = [col for col in required_columns if col not in products.columns]
    
    if missing_columns:
        raise ValueError(f"Thiếu các cột cần thiết trong product data: {missing_columns}")
    
    # Check for and handle nulls in product_id which is critical
    if products['product_id'].isnull().any():
        logger.error(f"Found {products['product_id'].isnull().sum()} null values in product_id column")
        logger.info("Removing rows with null product_id")
        products = products.dropna(subset=['product_id'])
    
    # Chuyển đổi product_id thành kiểu int nếu có thể, nếu không thì giữ nguyên dạng string
    if products['product_id'].dtype != 'int64':
        try:
            # Thử chuyển đổi sang int
            products['product_id'] = products['product_id'].astype(int)
        except (ValueError, TypeError):
            # Nếu không thể, để nguyên dạng, nhưng đảm bảo là string
            products['product_id'] = products['product_id'].astype(str)
            logger.info("Không thể chuyển product_id sang số nguyên, giữ nguyên dạng chuỗi.")
        
    return products

def load_event_data(filepath):
    """
    Load và xử lý dữ liệu sự kiện (events)
    """
    events = load_data(filepath)
    
    # Kiểm tra và đảm bảo cấu trúc cột
    expected_columns = ['user_id', 'event_type', 'product_id']
    if not all(col in events.columns for col in expected_columns):
        # Nếu không có tên cột chuẩn, giả định cấu trúc từ mẫu dữ liệu
        if events.shape[1] == 3:
            events.columns = expected_columns
        else:
            raise ValueError(f"Cấu trúc events data không đúng. Cần có các cột: {expected_columns}")
    
    # Chuyển đổi kiểu dữ liệu nếu cần
    events['user_id'] = events['user_id'].astype(int)
    events['product_id'] = events['product_id'].astype(int)
    
    # Xử lý giá trị event_type
    valid_event_types = ['home_page_view', 'detail_product_view', 'add_to_cart', 'purchase']
    if not events['event_type'].isin(valid_event_types).all():
        # Có thể cần map lại các giá trị không hợp lệ
        events['event_type'] = events['event_type'].apply(
            lambda x: x if x in valid_event_types else 'unknown_event'
        )
    
    # Tạo cột event_weight để định lượng tầm quan trọng của sự kiện
    event_weights = {
        'home_page_view': 1.0,
        'detail_product_view': 2.5,
        'add_to_cart': 4.0,
        'purchase': 5.0,
        'unknown_event': 0.5
    }
    events['event_weight'] = events['event_type'].map(event_weights)
    
    return events

def integrate_event_data(ratings, events):
    """
    Tích hợp dữ liệu events và ratings để có dataset đầy đủ hơn
    """
    # Tạo một bảng user-item từ sự kiện
    event_matrix = events.pivot_table(
        index='user_id',
        columns='product_id',
        values='event_weight',
        aggfunc='sum'
    ).reset_index()
    
    # Chuyển đổi từ ma trận sang format dài (long format)
    event_ratings = pd.melt(
        event_matrix, 
        id_vars=['user_id'],
        var_name='product_id',
        value_name='event_score'
    ).dropna()
    
    # Nếu đã có ratings, kết hợp với dữ liệu sự kiện
    if ratings is not None:
        # Đảm bảo kiểu dữ liệu phù hợp
        ratings['user_id'] = ratings['user_id'].astype(int)
        ratings['product_id'] = ratings['product_id'].astype(int)
        
        # Kết hợp với dữ liệu ratings (nếu có)
        combined = pd.merge(
            ratings,
            event_ratings,
            on=['user_id', 'product_id'],
            how='outer'
        )
        
        # Chuẩn hóa điểm event_score nếu có
        if 'event_score' in combined.columns:
            max_score = combined['event_score'].max()
            if max_score > 0:
                combined['event_score'] = combined['event_score'] / max_score * 5  # Giả sử thang điểm 5
            
            # Điền missing values
            combined['rating'] = combined['rating'].fillna(0)
            combined['event_score'] = combined['event_score'].fillna(0)
            
            # Tạo integrated_score kết hợp rating và event_score
            combined['integrated_score'] = 0.7 * combined['rating'] + 0.3 * combined['event_score']
        
        return combined
    else:
        # Nếu không có ratings, chỉ dùng event_ratings
        return event_ratings

def preprocess_data(ratings_path=None, products_path=None, events_path=None):
    """
    Hàm tổng thể để tiền xử lý và tích hợp tất cả dữ liệu
    """
    products = None
    ratings = None
    events = None
    
    # Load products (nếu có)
    if products_path and os.path.exists(products_path):
        products = load_product_details(products_path)
    
    # Load ratings (nếu có)
    if ratings_path and os.path.exists(ratings_path):
        ratings = load_data(ratings_path)
        ratings = clean_data(ratings)
    
    # Load events (nếu có)
    if events_path and os.path.exists(events_path):
        events = load_event_data(events_path)
    
    # Tích hợp dữ liệu
    if events is not None:
        integrated_data = integrate_event_data(ratings, events)
        
        # Kết hợp thông tin sản phẩm nếu có
        if products is not None:
            integrated_data = pd.merge(
                integrated_data,
                products,
                on='product_id',
                how='left'
            )
            # Đảm bảo không thiếu thông tin sản phẩm
            product_ids_missing = integrated_data[integrated_data['product_name'].isnull()]['product_id'].unique()
            if len(product_ids_missing) > 0:
                logger.warning(f"Có {len(product_ids_missing)} sản phẩm thiếu thông tin chi tiết. Bổ sung mặc định.")
                for pid in product_ids_missing:
                    integrated_data.loc[integrated_data['product_id'] == pid, 'product_name'] = f"Product {pid}"
                    integrated_data.loc[integrated_data['product_id'] == pid, 'product_type'] = 'unknown'
                    integrated_data.loc[integrated_data['product_id'] == pid, 'price'] = 0
                    integrated_data.loc[integrated_data['product_id'] == pid, 'rating'] = 0
        else:
            # Nếu không có products, tạo các trường mặc định
            for col in ['product_name', 'product_type', 'price', 'rating']:
                if col not in integrated_data.columns:
                    integrated_data[col] = None
            integrated_data['product_name'] = integrated_data['product_name'].fillna(integrated_data['product_id'].apply(lambda x: f"Product {x}"))
            integrated_data['product_type'] = integrated_data['product_type'].fillna('unknown')
            integrated_data['price'] = integrated_data['price'].fillna(0)
            integrated_data['rating'] = integrated_data['rating'].fillna(0)
        return integrated_data
    elif ratings is not None:
        # Nếu không có events, sử dụng ratings
        if products is not None:
            return pd.merge(ratings, products, on='product_id', how='left')
        else:
            return ratings
    else:
        # Trường hợp không có dữ liệu
        raise ValueError("Không có đủ dữ liệu để xử lý. Cần ít nhất ratings hoặc events.")