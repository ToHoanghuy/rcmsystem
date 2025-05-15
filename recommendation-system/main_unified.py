from flask import Flask, request, jsonify, redirect
import pandas as pd
import os
import sys
import numpy as np
import json
import uuid
from datetime import datetime
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_cors import CORS
from utils.json_utils import JSONEncoder, convert_to_json_serializable

# Thêm thư mục hiện tại vào đường dẫn Python để giải quyết việc import module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.preprocess import preprocess_data
from scripts.train_model import train_all_models
from scripts.generate_recommendations import generate_recommendations
from recommenders.collaborative import train_collaborative_model, train_advanced_collaborative_model, predict_for_cold_start
from recommenders.content_based import train_content_based_model, get_similar_products
from recommenders.hybrid import hybrid_recommendations, adaptive_hybrid_recommendations, switching_hybrid_recommendations
from recommenders.event_based import recommend_based_on_events
from recommenders.realtime import RealtimeRecommender
from utils.cache import RecommendationCache
from utils.realtime_integration import RealtimeIntegration
from utils.batch_processing import process_in_batches, batch_predict
from services.chat_service import ChatService

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Configure Flask to use custom JSON encoder for NumPy types
app.json_encoder = JSONEncoder

# Khởi tạo CORS để cho phép cross-origin requests
CORS(app, resources={r"/*": {"origins": "*"}})

# Khởi tạo SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", json=json)

# Khởi tạo cache
recommendation_cache = RecommendationCache(cache_expiry=3600, max_size=1000)

# Khởi tạo RealtimeRecommender
realtime_recommender = RealtimeRecommender()

# Khởi tạo services
chat_service = ChatService()

# Khởi tạo RealtimeIntegration
realtime_integration = None  # Sẽ được khởi tạo sau khi dữ liệu được tải

# Đường dẫn đến dữ liệu
DATA_DIR = os.path.join(current_dir, 'data')
RATINGS_PATH = os.path.join(DATA_DIR, 'raw', 'dataset.csv')
PRODUCTS_PATH = os.path.join(DATA_DIR, 'raw', 'products.csv')
EVENTS_PATH = os.path.join(DATA_DIR, 'raw', 'events.csv')

# Biến toàn cục để lưu mô hình và dữ liệu
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
    global data, product_details, collaborative_model, advanced_collaborative_model, content_model_data
    global transition_probs, testset, realtime_recommender, model_initialized, realtime_integration
    
    print("Initializing models...")
    
    # Xử lý dữ liệu
    try:
        print("Processing data...")
        data = preprocess_data(RATINGS_PATH, PRODUCTS_PATH, EVENTS_PATH)
        product_details = pd.read_csv(PRODUCTS_PATH)
        print(f"Loaded data with {len(data)} ratings and {len(product_details)} products")
        print(data.head())
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Huấn luyện mô hình
    try:
        print("Training models...")
        collaborative_model, content_similarity = train_all_models(data, product_details)
        content_model_data = content_similarity  # Đảm bảo tương thích
    except Exception as e:
        print(f"Error training models with train_all_models: {e}")
        # Thử phương pháp khác nếu có lỗi
        try:
            print("Falling back to individual model training...")
            collaborative_model, testset, predictions = train_collaborative_model(data, optimize_params=True)
            similarity_matrix, index_to_id, id_to_index = train_content_based_model(product_details)
            content_model_data = (similarity_matrix, index_to_id, id_to_index)
        except Exception as e:
            print(f"Error training individual models: {e}")
            return False
    
    # Khởi tạo thành công
    model_initialized = True
    
    # Khởi tạo realtime integration với models
    realtime_integration = RealtimeIntegration(
        socketio, 
        realtime_recommender,
        product_details=product_details,
        collaborative_model=collaborative_model,
        content_data=content_model_data
    )
    
    print("Models initialized successfully")
    return True

# Khởi tạo mô hình trước khi request đầu tiên
with app.app_context():
    initialize_models()

# API gợi ý
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
        if case == 'behavior_based':
            # Gợi ý dựa trên sự kiện người dùng
            recommendations = recommend_based_on_events(user_id, EVENTS_PATH, product_details)
            recommendations = recommendations.to_dict(orient='records')
        elif case == 'hybrid':
            if method == 'adaptive':
                recommendations = adaptive_hybrid_recommendations(
                    collaborative_model,
                    content_model_data,
                    user_id,
                    product_details,
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
                recommendations = hybrid_recommendations(
                    collaborative_model,
                    content_model_data,
                    user_id,
                    product_details
                )
            recommendations = recommendations.to_dict(orient='records')
            print(recommendations)
        elif case == 'collaborative':
            if product_id is None:
                # Dự đoán cho tất cả sản phẩm
                product_ids = product_details['product_id'].tolist()
                recommendations = []
                for product_id in product_ids:
                    prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                    recommendations.append({
                        "product_id": product_id,
                        "predicted_rating": prediction.est
                    })
            else:
                # Dự đoán cho một sản phẩm cụ thể
                prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                recommendations = {"product_id": product_id, "predicted_rating": prediction.est}
        elif case == 'content_based' and product_id is not None:
            # Xử lý dựa trên định dạng của content_model_data
            if isinstance(content_model_data, tuple):
                similarity_matrix, index_to_id, id_to_index = content_model_data
                similar_products = get_similar_products(
                    product_id,
                    similarity_matrix,
                    id_to_index,
                    index_to_id,
                    product_details,
                    top_n=10
                )
                recommendations = similar_products.to_dict(orient='records')
            else:
                recommendations = content_model_data[product_id].tolist()
        elif case == 'popular':
            # Lấy danh sách sản phẩm phổ biến
            top_products = product_details.sort_values(by='rating', ascending=False).head(10)
            recommendations = top_products.to_dict(orient='records')
        else:
            return jsonify({"error": f"Invalid case: {case}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      # Convert any NumPy types before caching and returning
    recommendations = convert_to_json_serializable(recommendations)
    
    # Lưu kết quả vào cache
    recommendation_cache.cache_recommendation(user_id, case, recommendations, product_id, method)
    
    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations,
        "from_cache": False,
        "case": case
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint cho chat service
    """
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
        
    try:
        reply = chat_service.process_query(user_message)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/event', methods=['POST'])
def record_event():
    """
    Record a user-product event and return real-time recommendations
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
            
    # Lấy dữ liệu từ request
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    event_type = data.get('event_type')
    
    if not all([user_id, product_id, event_type]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Ghi nhận sự kiện
    try:
        # Lấy thời gian hiện tại
        timestamp = datetime.now()
        
        # Ghi nhận sự kiện và gửi gợi ý thời gian thực 
        if realtime_integration:
            # record_event trong RealtimeIntegration trả về (success, recommendations)
            success, recommendations = realtime_integration.record_event(
                user_id, product_id, event_type, timestamp
            )
            # Nếu người dùng được đăng ký với socket thì đánh dấu là đã gửi qua socket
            sent_via_socket = user_id in realtime_integration.user_socket_connections
        else:
            # Sử dụng cách cũ nếu realtime_integration chưa sẵn sàng
            success = False
            if realtime_recommender:
                realtime_recommender.record_event(user_id, product_id, event_type, timestamp)
                success = True
                
            # Lấy gợi ý từ hybrid
            recommendations = hybrid_recommendations(
                collaborative_model=collaborative_model,
                content_model_data=content_model_data,
                user_id=user_id,
                product_details=product_details,
                top_n=5
            )
            sent_via_socket = False
          # Prepare recommendations for JSON serialization
        if isinstance(recommendations, pd.DataFrame):
            recs = recommendations.to_dict(orient='records')
        else:
            recs = recommendations
        
        # Convert any NumPy types to standard Python types
        recs = convert_to_json_serializable(recs)
        
        return jsonify({
            "status": "success", 
            "event_recorded": success,
            "recommendations": recs,
            "sent_via_socket": sent_via_socket
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/realtime-recommend', methods=['GET'])
def realtime_recommend():
    """
    Get real-time recommendations for a user
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
    
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    
    try:        # Lấy gợi ý thời gian thực
        if realtime_integration and realtime_integration.recommender is not None:
            try:
                recommendations = realtime_integration.get_recommendations(user_id, product_id, limit=10)
            except Exception as e:
                print(f"Error in realtime_integration.get_recommendations: {str(e)}")
                # Fallback to direct call
                recommendations = realtime_recommender.get_realtime_recommendations(
                    user_id=user_id,
                    current_product_id=product_id,
                    event_type=event_type,
                    product_details=product_details,
                    collaborative_model=collaborative_model,
                    content_data=content_model_data,
                    top_n=10
                )
        elif realtime_recommender is not None:
            # Sử dụng phương thức get_realtime_recommendations trực tiếp từ realtime_recommender
            recommendations = realtime_recommender.get_realtime_recommendations(
                user_id=user_id,
                current_product_id=product_id,
                event_type=event_type,
                product_details=product_details,
                collaborative_model=collaborative_model,
                content_data=content_model_data,
                top_n=10
            )
        else:
            # Sử dụng cách cũ nếu không có recommender nào sẵn sàng
            recommendations = hybrid_recommendations(
                collaborative_model=collaborative_model,
                content_model_data=content_model_data,
                user_id=user_id,
                product_details=product_details,
                top_n=5
            )        # Chuẩn hóa đầu ra
        if isinstance(recommendations, pd.DataFrame):
            recommendations = recommendations.to_dict(orient='records')
        elif not isinstance(recommendations, list) and recommendations is not None:
            recommendations = [recommendations]
            
        # Convert any NumPy types to standard Python types
        recommendations = convert_to_json_serializable(recommendations)
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations if recommendations is not None else []
        })
    except Exception as e:
        print(f"Error in realtime_recommend: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    API endpoint để đánh giá hiệu suất của các mô hình
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized, collaborative_model, content_model_data, testset, product_details
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    try:
        from models.evaluation import evaluate_model, evaluate_all_models, analyze_errors
        
        # Tạo dictionary chứa các mô hình
        models = {
            'collaborative': collaborative_model,
            'content_based': content_model_data
        }
        
        testsets = {}
        if testset:
            testsets['collaborative'] = testset
        
        # Đánh giá tất cả mô hình
        evaluation_report = evaluate_all_models(models, testsets, product_details)
        
        # Phân tích lỗi cho mô hình collaborative nếu có testset
        error_analysis = None
        if testset and collaborative_model:
            error_analysis = analyze_errors(collaborative_model, testset, product_details)
        
        return jsonify({
            'evaluation': evaluation_report,
            'error_analysis': error_analysis
        })
    except Exception as e:
        return jsonify({"error": f"Error evaluating models: {str(e)}"}), 500

@app.route('/admin/update-all-recommendations', methods=['POST'])
def update_all_recommendations():
    """
    Admin endpoint to update recommendations for all connected users
    """
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    global model_initialized, realtime_integration
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
    
    # Cập nhật gợi ý cho tất cả người dùng đang kết nối
    try:
        if realtime_integration:
            results = realtime_integration.update_recommendations_for_all_users()
            return jsonify({
                "status": "success",
                "user_count": len(results),
                "successful_updates": sum(1 for sent in results.values() if sent),
                "results": {str(uid): sent for uid, sent in results.items()}
            })
        else:
            return jsonify({"error": "Realtime integration not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/demo/realtime')
def realtime_demo():
    """
    Serve the real-time recommendation demo page
    """
    try:
        with open(os.path.join('static', 'realtime_demo.html'), 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error loading demo page: {str(e)}"

@app.route('/realtime')
def realtime_redirect():
    """
    Redirect to the real-time demo page
    """
    return redirect('/demo/realtime')

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """
    Handle Socket.IO connection
    """
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """
    Handle Socket.IO disconnection
    """
    if realtime_integration:
        realtime_integration.unregister_user(socket_id=request.sid)
    print(f"Client disconnected: {request.sid}")

@socketio.on('register_user')
def handle_register_user(data):
    """
    Register user with Socket.IO connection
    """
    user_id = data.get('user_id')
    if user_id and realtime_integration:
        success = realtime_integration.register_user(user_id, request.sid)
        if success:
            join_room(f'user_{user_id}')
            emit('registration_success', {'status': 'success', 'message': f'Registered as user {user_id}'})
        else:
            emit('registration_failed', {'status': 'error', 'message': 'Registration failed'})
    else:
        emit('registration_failed', {'status': 'error', 'message': 'No user_id provided or system not ready'})

# Điểm khởi chạy ứng dụng
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
