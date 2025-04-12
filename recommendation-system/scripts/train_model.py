from recommenders.collaborative import train_collaborative_model
from recommenders.content_based import train_content_based_model
from scripts.preprocess import preprocess_events


def train_all_models(data, product_details):
    # Huấn luyện Collaborative Filtering
    collaborative_model, testset = train_collaborative_model(data)
    
    # Huấn luyện Content-Based Filtering
    content_similarity = train_content_based_model(product_details)

    # Xử lý dữ liệu sự kiện (nếu cần)
    # events_summary = preprocess_events(events_path)
    
    return collaborative_model, content_similarity