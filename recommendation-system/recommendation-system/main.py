from flask import Flask, request, jsonify
import pandas as pd
import os
import sys
import numpy as np
import json
import uuid
from datetime import datetime
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_cors import CORS

# Add the parent directory to Python's path to resolve module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from preprocess.preprocess import preprocess, load_product_details, load_event_data
from recommenders.collaborative import train_collaborative_model, train_advanced_collaborative_model, predict_for_cold_start
from recommenders.content_based import train_content_based_model, get_similar_products
from recommenders.hybrid import hybrid_recommendations, adaptive_hybrid_recommendations, switching_hybrid_recommendations
from recommenders.event_based import recommend_based_on_events_advanced, analyze_user_behavior_sequence
from recommenders.realtime import RealtimeRecommender
from models.evaluation import evaluate_model, evaluate_all_models, analyze_errors
from utils.cache import RecommendationCache
from utils.realtime_integration import RealtimeIntegration
from utils.batch_processing import process_in_batches, batch_predict

app = Flask(__name__)
# Khởi tạo CORS để cho phép cross-origin requests
CORS(app, resources={r"/*": {"origins": "*"}})
# Khởi tạo SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo cache
recommendation_cache = RecommendationCache(cache_expiry=3600, max_size=1000)

# Khởi tạo đối tượng RealtimeRecommender
realtime_recommender = RealtimeRecommender()

# Khởi tạo đối tượng RealtimeIntegration để quản lý WebSocket và sự kiện thời gian thực
# Lưu ý: Các mô hình và dữ liệu chi tiết sẽ được cập nhật sau khi khởi tạo mô hình
realtime_integration = RealtimeIntegration(socketio, realtime_recommender)

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
    global data, product_details, collaborative_model, advanced_collaborative_model, content_model_data, transition_probs, testset, realtime_recommender, model_initialized
    
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
              # Khởi tạo thành công
        model_initialized = True
        
        # Cập nhật dữ liệu mô hình cho RealtimeIntegration
        realtime_integration.product_details = product_details
        realtime_integration.collaborative_model = collaborative_model
        realtime_integration.content_data = content_model_data
        
        # Thiết lập scheduled task để làm sạch các sự kiện hết hạn
        @socketio.on("before_server_stop")
        def cleanup_before_server_stop():
            """Clean up resources before server stops"""
            print("Server shutting down, cleaning up resources...")
            realtime_recommender._clean_expired_events()
            
        print("Models initialized successfully")
        return True
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return False
# Replace the deprecated @app.before_first_request decorator with the following approach
# Create a function that will run before the first request
with app.app_context():
    initialize_models()
    
# Scheduled task to clean expired events periodically
def clean_expired_events_task():
    """
    Task định kỳ để làm sạch các sự kiện hết hạn
    """
    with app.app_context():
        print("Running scheduled task: Cleaning expired events...")
        realtime_recommender._clean_expired_events()

def setup_scheduler(app):
    """
    Thiết lập scheduled tasks nếu thư viện cần thiết được cài đặt
    """
    try:
        from flask_apscheduler import APScheduler
        scheduler = APScheduler()
        scheduler.init_app(app)
        scheduler.add_job(id='clean_expired_events', func=clean_expired_events_task, 
                          trigger='interval', minutes=10)
        scheduler.start()
        print("Scheduled task for cleaning expired events has been set up")
        return True
    except ImportError:
        print("Flask-APScheduler not installed. Scheduled tasks are disabled.")
        print("To enable scheduled tasks, install flask-apscheduler with 'pip install flask-apscheduler'")
        return False

# Cố gắng thiết lập scheduled tasks
scheduler_active = setup_scheduler(app)

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

@app.route('/event', methods=['POST'])
def record_event():
    """
    API endpoint để ghi nhận sự kiện người dùng và trả về đề xuất thời gian thực
    
    POST parameters (JSON):
    - user_id: ID của người dùng
    - product_id: ID của sản phẩm
    - event_type: Loại sự kiện ('view', 'add_to_cart', 'purchase', etc.)
    - timestamp: Thời gian sự kiện (tùy chọn)
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
            
    # Lấy dữ liệu từ request
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    event_type = data.get('event_type')
    timestamp = data.get('timestamp')
    
    # Kiểm tra tham số bắt buộc
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not product_id:
        return jsonify({"error": "product_id is required"}), 400
    if not event_type:
        return jsonify({"error": "event_type is required"}), 400
        
    # Chuyển đổi timestamp nếu được cung cấp
    if timestamp:
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return jsonify({"error": "Invalid timestamp format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"}), 400
      # Ghi nhận sự kiện
    try:
        # Ghi nhận sự kiện và gửi gợi ý thời gian thực 
        success, recommendations, sent_via_socket = realtime_integration.record_event(
            user_id, product_id, event_type, timestamp
        )
        
        if not success:
            # Sử dụng cách cũ nếu đã lỗi
            realtime_recommender.record_event(user_id, product_id, event_type, timestamp)
            recommendations = get_realtime_recommendations(user_id, product_id, event_type)
            sent_via_socket = send_realtime_recommendation(user_id, recommendations)
        
        return jsonify({
            "status": "success", 
            "event_recorded": True,
            "recommendations": recommendations,
            "sent_via_socket": sent_via_socket
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/realtime-recommend', methods=['GET'])
def realtime_recommend():
    """
    API endpoint để lấy gợi ý thời gian thực cho người dùng
    
    Query parameters:
    - user_id: ID của người dùng
    - product_id: ID sản phẩm hiện tại (tùy chọn)
    - event_type: Loại sự kiện hiện tại (mặc định: 'view')
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    # Lấy các tham số từ request
    user_id = request.args.get('user_id', type=int)
    product_id = request.args.get('product_id', type=int)
    event_type = request.args.get('event_type', default='view', type=str)
    
    # Kiểm tra tham số bắt buộc
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    # Lấy gợi ý thời gian thực
    try:
        recommendations = get_realtime_recommendations(user_id, product_id, event_type)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_realtime_recommendations(user_id, current_product_id=None, event_type='view'):
    """
    Lấy gợi ý theo thời gian thực dựa trên sự kiện người dùng
    """
    # Lấy gợi ý từ recommender
    recommendations = realtime_recommender.get_realtime_recommendations(
        user_id=user_id,
        current_product_id=current_product_id,
        event_type=event_type,
        product_details=product_details,
        collaborative_model=collaborative_model,
        content_data=content_model_data,
        top_n=10
    )
    
    return recommendations

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """
    Xử lý kết nối socket mới
    """
    print(f'Client connected: {request.sid}')

@socketio.on('register_user')
def handle_register_user(data):
    """
    Đăng ký user_id cho kết nối socket hiện tại
    """
    user_id = data.get('user_id')
    if user_id:
        # Đăng ký user_id với kết nối socket hiện tại
        realtime_integration.register_user(user_id, request.sid)
        join_room(f'user_{user_id}')
        print(f'User {user_id} registered with socket ID {request.sid}')
        emit('registration_success', {'status': 'success', 'message': f'Registered as user {user_id}'})
    else:
        emit('registration_failed', {'status': 'error', 'message': 'No user_id provided'})

@socketio.on('disconnect')
def handle_disconnect():
    """
    Xử lý ngắt kết nối socket
    """
    # Hủy đăng ký tất cả người dùng đang sử dụng socket ID này
    realtime_integration.unregister_user(socket_id=request.sid)
    print(f'Client disconnected: {request.sid}')

def send_realtime_recommendation(user_id, recommendations):
    """
    Gửi gợi ý thời gian thực tới client qua socket
    """
    return realtime_integration.send_recommendations(user_id, recommendations)

@app.route('/admin/update-all-recommendations', methods=['POST'])
def update_all_recommendations():
    """
    Admin endpoint to update recommendations for all connected users
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    # Cập nhật gợi ý cho tất cả người dùng đang kết nối
    try:
        results = realtime_integration.update_recommendations_for_all_users()
        return jsonify({
            "status": "success",
            "user_count": len(results),
            "successful_updates": sum(1 for sent in results.values() if sent),
            "results": {str(uid): sent for uid, sent in results.items()}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/demo/realtime')
def realtime_demo():
    """
    Serve the real-time recommendation demo page
    """
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'realtime_demo.html'), 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)