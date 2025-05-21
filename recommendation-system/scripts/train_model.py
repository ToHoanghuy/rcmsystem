from recommenders.collaborative import train_collaborative_model
from recommenders.content_based import train_content_based_model
from recommenders.event_based import preprocess_events
import pandas as pd


def train_all_models(data, product_details, events_data=None):
    """
    Huấn luyện tất cả các mô hình đề xuất
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame chứa dữ liệu đánh giá (ratings)
    product_details : DataFrame
        DataFrame chứa thông tin chi tiết về sản phẩm/địa điểm
    events_data : DataFrame, optional
        DataFrame chứa dữ liệu sự kiện người dùng
        
    Returns:
    --------
    collaborative_model : object
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data : tuple
        Tuple chứa (similarity_matrix, index_to_id, id_to_index)
    """
    # Huấn luyện Collaborative Filtering
    collaborative_model, testset, predictions = train_collaborative_model(data)
    
    # Huấn luyện Content-Based Filtering
    content_model_data = train_content_based_model(product_details)
    
    # Xử lý dữ liệu sự kiện (nếu cần)
    events_summary = None
    if events_data is not None:
        events_summary = preprocess_events(events_data)
    
    print("Mô hình đã được huấn luyện thành công!")
    print(f"- Dữ liệu đánh giá: {len(data) if isinstance(data, pd.DataFrame) else 'N/A'} bản ghi")
    print(f"- Số lượng sản phẩm/địa điểm: {len(product_details)}")
    print(f"- Số lượng sự kiện: {len(events_data) if events_data is not None else 'N/A'}")
    
    return collaborative_model, content_model_data

def train_all_location_models(rating_data, location_details, events_data=None):
    """
    Huấn luyện các mô hình đề xuất cho dữ liệu location
    
    Parameters:
    -----------
    rating_data : DataFrame hoặc file path
        DataFrame hoặc đường dẫn đến file CSV chứa dữ liệu đánh giá
    location_details : DataFrame hoặc file path
        DataFrame hoặc đường dẫn đến file CSV chứa thông tin chi tiết về địa điểm
    events_data : DataFrame hoặc file path, optional
        DataFrame hoặc đường dẫn đến file CSV chứa dữ liệu sự kiện
        
    Returns:
    --------
    collaborative_model : object
        Mô hình Collaborative Filtering đã huấn luyện
    content_model_data : tuple
        Tuple chứa (similarity_matrix, index_to_id, id_to_index)    """
    
    # Nếu dữ liệu là đường dẫn đến file, đọc file
    if isinstance(rating_data, str):
        rating_data = pd.read_csv(rating_data)
    if isinstance(location_details, str):
        location_details = pd.read_csv(location_details)
    if isinstance(events_data, str) and events_data is not None:
        events_data = pd.read_csv(events_data)
    
    # Đảm bảo các cột cần thiết tồn tại trong location_details
    required_columns = ['product_id', 'name', 'description', 'category']
    if not all(col in location_details.columns for col in required_columns):
        print(f"WARNING: Dữ liệu địa điểm thiếu một số cột cần thiết: {required_columns}")
    
    # Đảm bảo cột province tồn tại, nếu không thêm cột giả
    if 'province' not in location_details.columns:
        print("WARNING: Dữ liệu địa điểm không có cột 'province', thêm cột giả")
        location_details['province'] = 'Unknown'
    
    print(f"Huấn luyện mô hình cho {len(location_details)} địa điểm...")
    
    # Gọi hàm train_all_models để huấn luyện các mô hình
    return train_all_models(rating_data, location_details, events_data)