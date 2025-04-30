from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import numpy as np

# Add the parent directory to Python's path to resolve module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from preprocess.preprocess import preprocess, load_product_details, load_event_data
from recommenders.collaborative import train_collaborative_model, train_advanced_collaborative_model, predict_for_cold_start
from recommenders.content_based import train_content_based_model, get_similar_products
from recommenders.hybrid import hybrid_recommendations, adaptive_hybrid_recommendations, switching_hybrid_recommendations
from recommenders.event_based import recommend_based_on_events_advanced, analyze_user_behavior_sequence
from models.evaluation import evaluate_model, evaluate_all_models, analyze_errors
from utils.cache import RecommendationCache
from utils.batch_processing import process_in_batches, batch_predict

app = Flask(__name__)

# Khởi tạo cache
recommendation_cache = RecommendationCache(cache_expiry=3600, max_size=1000)

# Đường dẫn đến dữ liệu
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'raw', 'dataset.csv')
PRODUCTS_PATH = os.path.join(DATA_DIR, 'raw', 'products.csv')
EVENTS_PATH = os.path.join(DATA_DIR, 'raw', 'events.csv')

# Biến global để lưu trữ mô hình và dữ liệu
data = None
product_details = None
collaborative_model = None
advanced_collaborative_model = None
content_model_data = None
transition_probs = None
testset = None
model_initialized = False

def initialize_models():
    """
    Khởi tạo và huấn luyện các mô hình
    """
    global data, product_details, collaborative_model, advanced_collaborative_model, content_model_data, transition_probs, testset, model_initialized
    
    print("Initializing models...")
    
    # Đọc và tiền xử lý dữ liệu
    try:
        data = preprocess(DATASET_PATH)
        product_details = load_product_details(PRODUCTS_PATH)
        print(f"Loaded data with {len(data)} ratings and {len(product_details)} products")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Huấn luyện mô hình Collaborative Filtering
    try:
        print("Training Collaborative Filtering model...")
        collaborative_model, testset, predictions = train_collaborative_model(data, optimize_params=True)
        
        # Đánh giá mô hình
        print("Evaluating Collaborative Filtering model...")
        metrics = evaluate_model(predictions)
    except Exception as e:
        print(f"Error training Collaborative Filtering model: {e}")
        # Tiếp tục với mô hình cơ bản nếu có lỗi
        try:
            collaborative_model, testset, predictions = train_collaborative_model(data, optimize_params=False)
        except Exception as e:
            print(f"Error training basic Collaborative Filtering model: {e}")
            return False
    
    # Huấn luyện mô hình Collaborative Filtering nâng cao (SVD++)
    try:
        print("Training Advanced Collaborative Filtering model (SVD++)...")
        advanced_collaborative_model, _, _ = train_advanced_collaborative_model(data)
    except Exception as e:
        print(f"Error training Advanced Collaborative Filtering model: {e}")
        # Không coi là lỗi nghiêm trọng, tiếp tục với mô hình cơ bản
    
    # Huấn luyện mô hình Content-Based Filtering
    try:
        print("Training Content-Based Filtering model...")
        similarity_matrix, index_to_id, id_to_index = train_content_based_model(product_details)
        content_model_data = (similarity_matrix, index_to_id, id_to_index)
    except Exception as e:
        print(f"Error training Content-Based Filtering model: {e}")
        return False
    
    # Phân tích chuỗi hành vi người dùng (nếu có dữ liệu sự kiện)
    try:
        if os.path.exists(EVENTS_PATH):
            print("Analyzing user behavior sequences...")
            transition_probs = analyze_user_behavior_sequence(EVENTS_PATH)
    except Exception as e:
        print(f"Error analyzing user behavior sequences: {e}")
        # Không coi là lỗi nghiêm trọng
    
    model_initialized = True
    print("Models initialized successfully!")
    return True

# Replace the deprecated @app.before_first_request decorator with the following approach
# Create a function that will run before the first request
with app.app_context():
    initialize_models()

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API endpoint cho việc gợi ý sản phẩm
    
    Query parameters:
    - user_id: ID của người dùng
    - case: Loại gợi ý (collaborative, content_based, hybrid, behavior_based)
    - product_id: ID của sản phẩm (chỉ cần cho content_based)
    - method: Phương pháp hybrid (adaptive, switching)
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    # Lấy các tham số từ request
    user_id = request.args.get('user_id', type=int)
    case = request.args.get('case', default='hybrid', type=str)
    product_id = request.args.get('product_id', type=int)
    method = request.args.get('method', default='adaptive', type=str)
    
    # Kiểm tra tham số
    if user_id is None and case not in ['content_based', 'popular']:
        return jsonify({"error": "user_id is required"}), 400
    
    if case == 'content_based' and product_id is None:
        return jsonify({"error": "product_id is required for content-based recommendations"}), 400
    
    # Kiểm tra cache
    cached_result = recommendation_cache.get_cached_recommendation(user_id, case, product_id, method)
    if cached_result:
        return jsonify({"user_id": user_id, "recommendations": cached_result, "from_cache": True})
    
    # Xử lý gợi ý dựa trên loại
    try:
        if case == 'collaborative':
            # Gợi ý dựa trên Collaborative Filtering
            recommendations = process_collaborative_recommendation(user_id, product_id)
        elif case == 'advanced_collaborative':
            # Gợi ý dựa trên Collaborative Filtering nâng cao (SVD++)
            if advanced_collaborative_model is None:
                return jsonify({"error": "Advanced collaborative model is not available"}), 400
            recommendations = process_advanced_collaborative_recommendation(user_id, product_id)
        elif case == 'content_based':
            # Gợi ý dựa trên Content-Based Filtering
            recommendations = process_content_based_recommendation(product_id)
        elif case == 'hybrid':
            # Gợi ý kết hợp
            recommendations = process_hybrid_recommendation(user_id, method)
        elif case == 'behavior_based':
            # Gợi ý dựa trên sự kiện người dùng
            recommendations = process_behavior_based_recommendation(user_id)
        elif case == 'popular':
            # Gợi ý sản phẩm phổ biến
            recommendations = process_popular_recommendation()
        else:
            return jsonify({"error": f"Invalid case: {case}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Lưu kết quả vào cache
    recommendation_cache.cache_recommendation(user_id, case, recommendations, product_id, method)
    
    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations,
        "from_cache": False,
        "case": case
    })

def process_collaborative_recommendation(user_id, product_id=None):
    """
    Xử lý gợi ý dựa trên Collaborative Filtering
    """
    if product_id is None:
        # Gợi ý tất cả sản phẩm cho người dùng
        product_ids = product_details['product_id'].tolist()
        recommendations = []
        
        # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(product_ids))
        predictions = batch_predict(collaborative_model, [user_id], product_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                recommendations.append({
                    "product_id": pid,
                    "predicted_rating": pred_rating
                })
            
            # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    else:
        # Dự đoán cho một sản phẩm cụ thể
        try:
            # Thử dự đoán với mô hình collaborative
            prediction = collaborative_model.predict(uid=user_id, iid=product_id)
            return {"product_id": product_id, "predicted_rating": prediction.est}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, collaborative_model, content_model_data[0], product_details)
            return {"product_id": product_id, "predicted_rating": pred_rating}

def process_advanced_collaborative_recommendation(user_id, product_id=None):
    """
    Xử lý gợi ý dựa trên Collaborative Filtering nâng cao (SVD++)
    """
    if product_id is None:
        # Gợi ý tất cả sản phẩm cho người dùng
        product_ids = product_details['product_id'].tolist()
        recommendations = []
        
        # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(product_ids))
        predictions = batch_predict(advanced_collaborative_model, [user_id], product_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                recommendations.append({
                    "product_id": pid,
                    "predicted_rating": pred_rating
                })
            
            # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    else:
        # Dự đoán cho một sản phẩm cụ thể
        try:
            # Thử dự đoán với mô hình advanced_collaborative
            prediction = advanced_collaborative_model.predict(uid=user_id, iid=product_id)
            return {"product_id": product_id, "predicted_rating": prediction.est}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, advanced_collaborative_model, content_model_data[0], product_details)
            return {"product_id": product_id, "predicted_rating": pred_rating}

def process_content_based_recommendation(product_id):
    """
    Xử lý gợi ý dựa trên Content-Based Filtering
    """
    similarity_matrix, index_to_id, id_to_index = content_model_data
    
    # Lấy sản phẩm tương tự dựa trên content-based
    similar_products = get_similar_products(
        product_id,
        similarity_matrix,
        id_to_index,
        index_to_id,
        product_details,
        top_n=10
    )
    
    return similar_products.to_dict(orient='records')

def process_hybrid_recommendation(user_id, method='adaptive'):
    """
    Xử lý gợi ý kết hợp
    """
    # Đọc dữ liệu sự kiện nếu có
    user_events = None
    if os.path.exists(EVENTS_PATH):
        try:
            event_data = load_event_data(EVENTS_PATH)
            # Tiền xử lý dữ liệu sự kiện
            if 'user_id' in event_data.columns and 'event_type' in event_data.columns:
                user_events = event_data.groupby('user_id').sum()
        except Exception as e:
            print(f"Error loading event data: {e}")
    
    # Gọi hàm gợi ý kết hợp dựa trên phương pháp
    if method == 'adaptive':
        recommendations = adaptive_hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            product_details,
            user_events=user_events,
            top_n=10
        )
    elif method == 'switching':
        recommendations, _ = switching_hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            product_details,
            confidence_threshold=0.8,
            top_n=10
        )
    else:
        # Fallback to default
        recommendations = hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            product_details,
            method="adaptive",
            top_n=10,
            user_events=user_events
        )
    
    return recommendations.to_dict(orient='records')

def process_behavior_based_recommendation(user_id):
    """
    Xử lý gợi ý dựa trên sự kiện người dùng
    """
    if not os.path.exists(EVENTS_PATH):
        return jsonify({"error": "Events data not available"}), 404
    
    # Gọi hàm gợi ý nâng cao dựa trên sự kiện
    recommendations = recommend_based_on_events_advanced(
        user_id,
        EVENTS_PATH,
        product_details,
        transition_probs=transition_probs,
        top_n=10
    )
    
    return recommendations.to_dict(orient='records')

def process_popular_recommendation():
    """
    Lấy danh sách sản phẩm phổ biến
    """
    # Đơn giản là sắp xếp theo đánh giá trung bình
    top_products = product_details.sort_values(by='rating', ascending=False).head(10)
    
    return top_products.to_dict(orient='records')

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    API endpoint để đánh giá hiệu suất của các mô hình
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    # Tạo dictionary chứa các mô hình
    models = {
        'collaborative': collaborative_model,
        'content_based': content_model_data,
        'hybrid': None,  # No hybrid model to evaluate directly
        'event_based': None  # No event-based model to evaluate directly
    }
    
    testsets = {
        'collaborative': testset
    }
    
    # Đánh giá tất cả mô hình
    evaluation_report = evaluate_all_models(models, testsets, product_details)
    
    # Phân tích lỗi cho mô hình collaborative
    error_analysis = analyze_errors(collaborative_model, testset, product_details)
    
    return jsonify({
        'evaluation': evaluation_report,
        'error_analysis': error_analysis
    })

@app.route('/cache/stats', methods=['GET'])
def cache_stats():
    """
    API endpoint để lấy thông tin thống kê về cache
    """
    return jsonify(recommendation_cache.get_cache_stats())

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """
    API endpoint để xóa cache
    """
    recommendation_cache.clear_cache()
    return jsonify({"status": "Cache cleared"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)