from flask import Flask, request, jsonify
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
from services.mongodb_monitor import MongoDBChangeMonitor
from services.data_importer import DataImporter
from utils.mongodb_loader import load_ratings_from_mongodb, load_events_from_json, load_mongodb_locations
from utils.mongodb_store import MongoDBEventStore
from utils.json_utils import clean_json_data, convert_to_json_serializable
from utils.mongodb_json_cleaner import clean_mongodb_json

app = Flask(__name__)
# Khởi tạo CORS để cho phép cross-origin requests
CORS(app, resources={r"/*": {"origins": "*"}})
# Khởi tạo SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Patch RealtimeRecommender để thêm các phương thức mới
from utils.realtime_patch import patch_realtime_recommender
patch_realtime_recommender()

# Khởi tạo cache
recommendation_cache = RecommendationCache(cache_expiry=3600, max_size=1000)

# Khởi tạo đối tượng RealtimeRecommender
# Tạo sau khi đã patch để đảm bảo các phương thức mới được áp dụng
realtime_recommender = RealtimeRecommender()
# Đảm bảo các thuộc tính cần thiết được khởi tạo
realtime_recommender.recent_events = {}
realtime_recommender.recent_ratings = {}

# Đăng ký API endpoints
from services.api_endpoints import register_api_endpoints

# Khởi tạo đối tượng RealtimeIntegration để quản lý WebSocket và sự kiện thời gian thực
# Lưu ý: Các mô hình và dữ liệu chi tiết sẽ được cập nhật sau khi khởi tạo mô hình
realtime_integration = RealtimeIntegration(socketio, realtime_recommender)

# Đường dẫn đến dữ liệu
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Đăng ký API endpoints để tích hợp với frontend
register_api_endpoints(app, realtime_recommender, realtime_integration, recommendation_cache, DATA_DIR=DATA_DIR)
# Sử dụng dữ liệu location thay vì sản phẩm
DATASET_PATH = os.path.join(DATA_DIR, 'raw', 'location_ratings.csv')
PRODUCTS_PATH = os.path.join(DATA_DIR, 'raw', 'locations.csv')
EVENTS_PATH = os.path.join(DATA_DIR, 'raw', 'location_events.csv')

# Biến global để lưu trữ mô hình và dữ liệu
data = None
location_details = None  # Đổi tên từ product_details thành location_details
collaborative_model = None
advanced_collaborative_model = None
content_model_data = None
transition_probs = None
testset = None
model_initialized = False
mongodb_monitor = None  # Thêm biến toàn cục để quản lý MongoDB monitor
data_importer = None  # Thêm biến toàn cục để quản lý việc nhập dữ liệu

def reinitialize_models():
    """
    Cập nhật lại các mô hình khi có dữ liệu mới từ MongoDB
    """
    global data, location_details, collaborative_model, advanced_collaborative_model, content_model_data
    
    try:
        print("Reinitializing models due to MongoDB data changes...")
        
        # Reload location details từ file đã được cập nhật bởi mongodb_monitor
        location_details = load_product_details(PRODUCTS_PATH)
        print(f"Reloaded location details: {len(location_details)} locations")
        
        # Cập nhật mô hình Content-Based Filtering
        print("Retraining Content-Based Filtering model...")
        similarity_matrix, index_to_id, id_to_index = train_content_based_model(location_details)
        content_model_data = (similarity_matrix, index_to_id, id_to_index)
        
        # Nếu có dữ liệu ratings mới, ta cũng có thể cập nhật mô hình collaborative
        # Nhưng việc này đòi hỏi dữ liệu ratings được cập nhật, không chỉ location data
        # Để đơn giản, hiện tại ta chỉ cập nhật content-based model
        
        # Cập nhật dữ liệu mô hình cho RealtimeIntegration
        if 'realtime_integration' in globals() and realtime_integration is not None:
            realtime_integration.product_details = location_details
            realtime_integration.content_data = content_model_data
        
        # Làm sạch cache để đảm bảo đề xuất mới sẽ được tính toán
        if 'recommendation_cache' in globals() and recommendation_cache is not None:
            recommendation_cache.clear()
        
        print("Model reinitialization completed successfully")
        return True
    except Exception as e:
        print(f"Error during model reinitialization: {e}")
        return False

def initialize_models():
    """
    Khởi tạo và huấn luyện các mô hình với dữ liệu thật từ MongoDB và JSON
    """
    global data, location_details, collaborative_model, advanced_collaborative_model, content_model_data
    global transition_probs, testset, realtime_recommender, model_initialized, mongodb_monitor
    
    print("Initializing models for location recommendations...")
    
    # Đọc và tiền xử lý dữ liệu
    try:
        # Dùng dữ liệu thật từ MongoDB và JSON
        data = load_ratings_from_mongodb()  # Lấy rating từ collection Review
        location_details = load_mongodb_locations()  # Lấy location từ MongoDB
        print(f"Loaded data with {len(data)} ratings and {len(location_details)} locations (from MongoDB)")
        print(f"Ratings columns: {list(data.columns)}")
        print(f"Ratings sample:\n{data.head(3)}")
        required_cols = {"user_id", "location_id", "rating", "timestamp"}
        if data.empty or not required_cols.issubset(set(data.columns)):
            print(f"ERROR: MongoDB ratings data is missing or malformed. Required columns: {required_cols}. Found: {list(data.columns)}")
            print("Please ensure your Review collection in MongoDB is populated with user_id, location_id, rating, timestamp.")
            return False
    except Exception as e:
        print(f"Error loading location data: {e}")
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
        similarity_matrix, index_to_id, id_to_index = train_content_based_model(location_details)
        content_model_data = (similarity_matrix, index_to_id, id_to_index)
    except Exception as e:
        print(f"Error training Content-Based Filtering model: {e}")
        return False
    
    # Phân tích chuỗi hành vi người dùng (nếu có dữ liệu sự kiện)
    try:
        # Dùng dữ liệu sự kiện thật từ JSON
        events_data = load_events_from_json()
        if not events_data.empty:
            print(f"Loaded {len(events_data)} real events from JSON")
            # Nếu cần phân tích hành vi, truyền events_data vào
            transition_probs = analyze_user_behavior_sequence(events_data)
        
        # Khởi tạo thành công
        model_initialized = True
        # Cập nhật dữ liệu mô hình cho RealtimeIntegration
        realtime_integration.product_details = location_details  # Sử dụng location_details
        realtime_integration.collaborative_model = collaborative_model
        realtime_integration.content_data = content_model_data
        
        # Thiết lập scheduled task để làm sạch các sự kiện hết hạn
        @socketio.on("before_server_stop")
        def cleanup_before_server_stop():
            """Clean up resources before server stops"""
            print("Server shutting down, cleaning up resources...")
            realtime_recommender._clean_expired_events()
            # Dừng MongoDB monitor nếu đang hoạt động
            if mongodb_monitor:
                mongodb_monitor.stop()
          # Khởi tạo MongoDB monitor để theo dõi thay đổi trong dữ liệu địa điểm
        print("Initializing MongoDB Change Stream Monitor...")
        global data_importer
        mongodb_monitor = MongoDBChangeMonitor(update_callback=reinitialize_models, debounce_seconds=300)
        mongodb_monitor.start()
        
        # Khởi tạo Data Importer
        data_importer = DataImporter(update_callback=reinitialize_models)
        print("MongoDB Change Stream Monitor and Data Importer started successfully")
            
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

@app.route('/track_legacy', methods=['POST'])
def track_event_legacy():
    """
    Legacy API endpoint để thu thập sự kiện người dùng từ frontend
    Giữ lại cho khả năng tương thích ngược, khuyến nghị sử dụng /api/track thay thế
    """
    try:
        event_data = request.json
        if not event_data or not isinstance(event_data, dict):
            return jsonify({"success": False, "error": "Invalid event data format"}), 400
        required_fields = ["user_id", "event_type", "location_id"]
        for field in required_fields:
            if field not in event_data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now().isoformat()
        if "data" not in event_data:
            event_data["data"] = {}
        # Lưu sự kiện vào MongoDB (collection events)
        event_store = MongoDBEventStore()
        mongo_event = {
            'user_id': str(event_data['user_id']),
            'location_id': str(event_data['location_id']),
            'event_type': event_data['event_type'],
            'timestamp': event_data['timestamp'],
            'data': event_data['data']
        }
        success = event_store.store_event(mongo_event)
        if not success:
            return jsonify({"success": False, "error": "Failed to store event in MongoDB"}), 500
        return jsonify({"success": True, "message": "Event tracked successfully (legacy endpoint)"}), 200
    except Exception as e:
        print(f"Error tracking event (legacy): {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/recommend_legacy', methods=['GET'])
def recommend_legacy():
    """
    Legacy API endpoint cho việc gợi ý sản phẩm
    Giữ lại cho khả năng tương thích ngược, khuyến nghị sử dụng /api/recommend thay thế
    
    Query parameters:
    - user_id: ID của người dùng
    - case: Loại gợi ý (collaborative, content_based, hybrid, behavior_based)
    - product_id: ID của sản phẩm (chỉ cần cho content_based)
    - method: Phương pháp hybrid (adaptive, switching)
    """
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500    
    user_id = request.args.get('user_id')
    case = request.args.get('case', default='hybrid', type=str)
    product_id = request.args.get('product_id')
    method = request.args.get('method', default='adaptive', type=str)
    
    # Log request params for debugging
    print(f"\n==== RECOMMEND LEGACY REQUEST ====")
    print(f"user_id: {user_id}, type: {type(user_id)}")
    print(f"case: {case}, type: {type(case)}")
    print(f"product_id: {product_id}, type: {type(product_id)}")
    print(f"method: {method}, type: {type(method)}")
    
    # Kiểm tra xem product_id có khả năng là MongoDB ObjectID
    if product_id and isinstance(product_id, str) and len(product_id) == 24 and all(c in '0123456789abcdef' for c in product_id.lower()):
        print(f"product_id appears to be a MongoDB ObjectID: {product_id}")
    
    print(f"====================================\n")
    
    if user_id is None and case not in ['content_based', 'popular']:
        return jsonify({"error": "user_id is required"}), 400
    if case == 'content_based' and product_id is None:
        return jsonify({"error": "product_id is required for content-based recommendations"}), 400
      # Kiểm tra cache
    cached_result = recommendation_cache.get_cached_recommendation(user_id, case, product_id, method)
    if cached_result:
        print(f"Using cached result for user_id={user_id}, case={case}, product_id={product_id}")        # Làm sạch dữ liệu từ cache trước khi trả về - sử dụng hàm nâng cao
        cleaned_cached_result = clean_mongodb_json(cached_result)
        return jsonify({"user_id": user_id, "recommendations": cleaned_cached_result, "from_cache": True})
    
    try:
        if case == 'collaborative':
            recommendations = process_collaborative_recommendation(user_id, product_id)
        elif case == 'advanced_collaborative':
            if advanced_collaborative_model is None:
                return jsonify({"error": "Advanced collaborative model is not available"}), 400
            recommendations = process_advanced_collaborative_recommendation(user_id, product_id)
        elif case == 'content_based':
            recommendations = process_content_based_recommendation(product_id)
        elif case == 'hybrid':
            recommendations = process_hybrid_recommendation(user_id, method)
        elif case == 'behavior_based':
            recommendations = process_behavior_based_recommendation(user_id)
        elif case == 'popular':
            recommendations = process_popular_recommendation()
        else:
            return jsonify({"error": f"Invalid case: {case}"}), 400
        
        # Kiểm tra xem recommendations có phải là danh sách rỗng không
        if not recommendations:
            print(f"Warning: Empty recommendations for case={case}, product_id={product_id}")
            if case == 'content_based':
                print("Returning popular recommendations as fallback")
                recommendations = process_popular_recommendation()
    except Exception as e:
        error_msg = f"Error processing recommendation: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500
    
    # Lưu vào cache
    recommendation_cache.cache_recommendation(user_id, case, recommendations, product_id, method)
    
    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations,
        "from_cache": False,
        "case": case
    })
    #     "from_cache": False,
    #     "case": case
    # })

@app.route('/recommend_legacy/', methods=['GET'])
def recommend_legacy_with_slash():
    """
    Legacy API endpoint cho việc gợi ý sản phẩm (version với dấu gạch chéo ở cuối)
    Sử dụng cùng một hàm xử lý với endpoint không có dấu gạch chéo ở cuối
    """
    return recommend_legacy()

def process_collaborative_recommendation(user_id, product_id=None):
    """
    Xử lý gợi ý dựa trên Collaborative Filtering
    """
    if product_id is None:
        # Gợi ý tất cả địa điểm cho người dùng
        location_ids = location_details['product_id'].tolist()
        recommendations = []
        
        # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(location_ids))
        predictions = batch_predict(collaborative_model, [user_id], location_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                # Lấy thông tin chi tiết của location từ location_details
                location_info = location_details[location_details['product_id'] == pid].to_dict('records')
                
                if location_info:
                    # Kết hợp thông tin chi tiết với dự đoán rating
                    location_data = location_info[0].copy()
                    location_data["predicted_rating"] = pred_rating
                    recommendations.append(location_data)
                else:
                    # Nếu không tìm thấy thông tin chi tiết, chỉ trả về ID và rating
                    recommendations.append({
                        "product_id": pid,
                        "predicted_rating": pred_rating
                    })
              # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations
    else:
        # Dự đoán cho một địa điểm cụ thể
        try:
            # Thử dự đoán với mô hình collaborative
            prediction = collaborative_model.predict(uid=user_id, iid=product_id)
            pred_rating = prediction.est
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, collaborative_model, content_model_data[0], location_details)
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}

def process_advanced_collaborative_recommendation(user_id, product_id=None):
    """
    Xử lý gợi ý dựa trên Collaborative Filtering nâng cao (SVD++)
    """
    # Đảm bảo user_id có định dạng nhất quán (string hoặc số)
    if user_id is not None and not isinstance(user_id, (int, str)):
        user_id = str(user_id)
        
    if product_id is None:
        # Gợi ý tất cả địa điểm cho người dùng
        location_ids = location_details['product_id'].tolist()
        recommendations = []
        
        # Đảm bảo location_ids có định dạng nhất quán
        location_ids = [str(id) if isinstance(id, str) else id for id in location_ids]
          # Sử dụng batch predict để tăng hiệu suất
        batch_size = min(100, len(location_ids))
        predictions = batch_predict(advanced_collaborative_model, [user_id], location_ids, batch_size)
        
        if user_id in predictions:
            user_predictions = predictions[user_id]
            for pid, pred_rating in user_predictions.items():
                # Lấy thông tin chi tiết của location từ location_details
                location_info = location_details[location_details['product_id'] == pid].to_dict('records')
                
                if location_info:
                    # Kết hợp thông tin chi tiết với dự đoán rating
                    location_data = location_info[0].copy()
                    location_data["predicted_rating"] = pred_rating
                    recommendations.append(location_data)
                else:
                    # Nếu không tìm thấy thông tin chi tiết, chỉ trả về ID và rating
                    recommendations.append({
                        "product_id": pid,
                        "predicted_rating": pred_rating
                    })
            
            # Sắp xếp theo điểm dự đoán giảm dần
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations
    else:
        # Dự đoán cho một địa điểm cụ thể
        try:
            # Thử dự đoán với mô hình advanced_collaborative
            prediction = advanced_collaborative_model.predict(uid=user_id, iid=product_id)
            pred_rating = prediction.est
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}
        except:
            # Nếu không được, sử dụng xử lý cold-start
            pred_rating = predict_for_cold_start(user_id, product_id, advanced_collaborative_model, content_model_data[0], location_details)
            
            # Lấy thông tin chi tiết của location
            location_info = location_details[location_details['product_id'] == product_id].to_dict('records')
            if location_info:
                # Kết hợp thông tin chi tiết với dự đoán rating
                result = location_info[0].copy()
                result["predicted_rating"] = pred_rating
                return result
            return {"product_id": product_id, "predicted_rating": pred_rating}

def process_content_based_recommendation(product_id):
    """
    Xử lý gợi ý dựa trên Content-Based Filtering
    """
    similarity_matrix, index_to_id, id_to_index = content_model_data
    
    # In thông tin debug để phân tích vấn đề
    print(f"Processing content-based recommendation for product_id: {product_id}, type: {type(product_id)}")
    print(f"id_to_index contains {len(id_to_index)} keys, first 5 keys: {list(id_to_index.keys())[:5]}")
    
    # Kiểm tra format của product_id - chuyển đổi nếu cần thiết
    if product_id not in id_to_index:
        print(f"Product ID {product_id} not found directly in id_to_index")
        
        # Thử với dạng chuỗi
        if str(product_id) in id_to_index:
            product_id = str(product_id)
            print(f"Converted product_id to string: {product_id}")
        # Chỉ thử chuyển đổi sang int nếu chuỗi toàn số
        elif isinstance(product_id, str) and product_id.isdigit() and int(product_id) in id_to_index:
            product_id = int(product_id)
            print(f"Converted product_id to int: {product_id}")
        # Thử tìm kiếm khớp một phần nếu là MongoDB ObjectID
        else:
            found = False
            print(f"Searching for string match for product_id: {product_id}")
            
            # In ra một số khóa để debug
            sample_keys = list(id_to_index.keys())[:10]
            print(f"Sample keys in id_to_index: {sample_keys}")
            
            # Kiểm tra xem khóa có phải MongoDB ObjectID hay không
            is_mongo_id = isinstance(product_id, str) and len(product_id) == 24 and all(c in '0123456789abcdef' for c in product_id.lower())
            if is_mongo_id:
                print(f"Product ID appears to be a MongoDB ObjectID: {product_id}")
                
                # Tiếp cận 1: Tìm khớp chuỗi chính xác
                for idx in id_to_index.keys():
                    if str(idx) == str(product_id):
                        product_id = idx
                        found = True
                        print(f"Found exact string match for MongoDB ObjectID: {idx}")
                        break
                
                # Tiếp cận 2: Tìm khớp phần cuối của ObjectID
                if not found:
                    print("Trying to match the last part of ObjectID")
                    for idx in id_to_index.keys():
                        # Kiểm tra xem idx có phải là một chuỗi không
                        if isinstance(idx, str):
                            # Nếu cả hai đều là ObjectID, kiểm tra xem có phần trùng nhau không
                            if len(idx) >= 8 and len(product_id) >= 8 and idx[-8:] == product_id[-8:]:
                                product_id = idx
                                found = True
                                print(f"Found partial match for ObjectID suffix: {idx}")
                                break
            else:
                # Không phải MongoDB ObjectID, thử so khớp chuỗi thông thường
                for idx in id_to_index.keys():
                    if str(idx) == str(product_id):
                        product_id = idx
                        found = True
                        print(f"Found matching key in id_to_index: {idx}")
                        break
                        
            if not found:
                print(f"Warning: product_id {product_id} not found in id_to_index")
                # Cung cấp các địa điểm phổ biến thay vì danh sách trống
                print(f"Returning popular locations instead of empty list")
                return process_popular_recommendation()
    
    # Lấy địa điểm tương tự dựa trên content-based
    try:
        print(f"Calling get_similar_products with product_id: {product_id}")
        similar_locations = get_similar_products(
            product_id,
            similarity_matrix,
            id_to_index,
            index_to_id,
            location_details,
            top_n=10
        )
        print(f"Found {len(similar_locations)} similar locations")
        
        # Kiểm tra xem kết quả có rỗng không
        if similar_locations.empty:
            print(f"Warning: No similar locations found for product_id {product_id}")
            print(f"Returning popular locations instead of empty list")
            return process_popular_recommendation()
            
    except Exception as e:
        print(f"Error getting similar products: {e}")
        print(f"Returning popular locations as fallback due to error")
        return process_popular_recommendation()
        import traceback
        traceback.print_exc()
        return []    
    except Exception as e:
        print(f"Error getting similar products: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return similar_locations.to_dict(orient='records')

def process_hybrid_recommendation(user_id, method='adaptive'):
    """
    Xử lý gợi ý kết hợp
    """
    # Đọc dữ liệu sự kiện nếu có
    user_events = None
    try:
        event_data = load_events_from_json()
        if not event_data.empty and 'user_id' in event_data.columns and 'event_type' in event_data.columns:
            user_events = event_data.groupby('user_id').sum()
    except Exception as e:
        print(f"Error loading event data: {e}")
    except Exception as e:
        print(f"Error loading event data: {e}")
    # Gọi hàm gợi ý kết hợp dựa trên phương pháp
    if method == 'adaptive':
        recommendations = adaptive_hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            location_details,
            user_events=user_events,
            top_n=10
        )
    elif method == 'switching':
        recommendations, _ = switching_hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            location_details,
            confidence_threshold=0.8,
            top_n=10
        )
    else:
        recommendations = hybrid_recommendations(
            collaborative_model,
            content_model_data,
            user_id,
            location_details,
            method="adaptive",
            top_n=10,
            user_events=user_events
        )
    return recommendations.to_dict(orient='records')

def process_behavior_based_recommendation(user_id):
    """
    Xử lý gợi ý dựa trên sự kiện người dùng
    """
    events_data = load_events_from_json()
    if events_data.empty:
        return jsonify({"error": "Events data not available"}), 404
    recommendations = recommend_based_on_events_advanced(
        user_id,
        events_data,
        location_details,
        transition_probs=transition_probs,
        top_n=10
    )
    return recommendations.to_dict(orient='records')

def process_popular_recommendation():
    """
    Lấy danh sách sản phẩm phổ biến
    """
    # Đơn giản là sắp xếp theo đánh giá trung bình
    top_products = location_details.sort_values(by='rating', ascending=False).head(10)
    
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
    evaluation_report = evaluate_all_models(models, testsets, location_details)
    
    # Phân tích lỗi cho mô hình collaborative
    error_analysis = analyze_errors(collaborative_model, testset, location_details)
    
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
    """
    global model_initialized
    if not model_initialized:
        if not initialize_models():
            return jsonify({"error": "Failed to initialize models"}), 500
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
    # Ghi nhận sự kiện vào MongoDB
    try:
        event_store = MongoDBEventStore()
        event_data = {
            'user_id': str(user_id),
            'location_id': str(product_id),
            'event_type': event_type,
            'timestamp': timestamp if timestamp else datetime.now(),
            'data': data.get('data', {})
        }
        success = event_store.store_event(event_data)
        if not success:
            return jsonify({"error": "Failed to store event in MongoDB"}), 500        
        # Gửi gợi ý thời gian thực
        recommendations = get_realtime_recommendations(user_id, product_id, event_type)
        sent_via_socket = send_realtime_recommendation(user_id, recommendations)
        
        # Làm sạch dữ liệu JSON trước khi trả về - sử dụng hàm nâng cao
        cleaned_recommendations = clean_mongodb_json(recommendations)
        
        return jsonify({
            "status": "success",
            "event_recorded": True,
            "recommendations": cleaned_recommendations,
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
    
    # Lấy các tham số từ request, không chỉ định kiểu để xử lý MongoDB ObjectID
    user_id = request.args.get('user_id')
    product_id = request.args.get('product_id')
    event_type = request.args.get('event_type', default='view', type=str)
    
    # In log để debug
    print(f"\n==== REALTIME RECOMMEND REQUEST ====")
    print(f"user_id: {user_id}, type: {type(user_id)}")
    print(f"product_id: {product_id}, type: {type(product_id)}")
    print(f"event_type: {event_type}")
    print(f"====================================\n")
    
    # Kiểm tra tham số bắt buộc
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    # Kiểm tra xem user_id có khả năng là MongoDB ObjectID
    is_mongo_id = isinstance(user_id, str) and len(user_id) == 24 and all(c in '0123456789abcdef' for c in user_id.lower())
    if is_mongo_id:
        print(f"user_id appears to be a MongoDB ObjectID: {user_id}")
    elif user_id.isdigit():
        # Nếu là chuỗi số nguyên, chuyển đổi thành số
        user_id = int(user_id)
        print(f"Converted user_id to integer: {user_id}")
    
    # Xử lý tương tự cho product_id nếu có
    if product_id:
        is_product_mongo_id = isinstance(product_id, str) and len(product_id) == 24 and all(c in '0123456789abcdef' for c in product_id.lower())
        if is_product_mongo_id:
            print(f"product_id appears to be a MongoDB ObjectID: {product_id}")
        elif product_id.isdigit():
            product_id = int(product_id)
            print(f"Converted product_id to integer: {product_id}")
      # Lấy gợi ý thời gian thực
    try:
        recommendations = get_realtime_recommendations(user_id, product_id, event_type)
        
        # Làm sạch dữ liệu trước khi trả về JSON - sử dụng hàm nâng cao
        cleaned_recommendations = clean_mongodb_json(recommendations)
        
        return jsonify({
            "user_id": user_id,
            "recommendations": cleaned_recommendations
        })
    except Exception as e:
        print(f"Error in realtime_recommend: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "Failed to generate realtime recommendations. Please check the server logs for details."
        }), 500

def get_realtime_recommendations(user_id, current_product_id=None, event_type='view'):
    """
    Lấy gợi ý theo thời gian thực dựa trên sự kiện người dùng
    """
    # Log the parameters for debugging
    print(f"GET_REALTIME_RECOMMENDATIONS - user_id: {user_id}, type: {type(user_id)}")
    print(f"GET_REALTIME_RECOMMENDATIONS - current_product_id: {current_product_id}, type: {type(current_product_id)}")
    print(f"GET_REALTIME_RECOMMENDATIONS - event_type: {event_type}")
    
    # Log information about location_details
    if location_details is not None:
        print(f"GET_REALTIME_RECOMMENDATIONS - location_details has {len(location_details)} rows")
        print(f"GET_REALTIME_RECOMMENDATIONS - location_details columns: {list(location_details.columns)}")
        
        # Check if there's an ID column that could be used
        possible_id_columns = ['location_id', 'product_id', 'id']
        id_column = next((col for col in possible_id_columns if col in location_details.columns), None)
        if id_column:
            print(f"GET_REALTIME_RECOMMENDATIONS - Using '{id_column}' as the ID column")
        else:
            print(f"GET_REALTIME_RECOMMENDATIONS - Warning: No standard ID column found!")
    else:
        print(f"GET_REALTIME_RECOMMENDATIONS - Warning: location_details is None!")
      # Lấy gợi ý từ recommender
    recommendations = realtime_recommender.get_realtime_recommendations(
        user_id=user_id,
        current_product_id=current_product_id,
        event_type=event_type,
        product_details=location_details,
        collaborative_model=collaborative_model,
        content_data=content_model_data,
        top_n=10
    )
    
    # Log the number of recommendations for debugging
    print(f"GET_REALTIME_RECOMMENDATIONS - Found {len(recommendations)} recommendations")
    
    # If no recommendations were found, fall back to popular recommendations
    if not recommendations:
        print(f"GET_REALTIME_RECOMMENDATIONS - No recommendations found, falling back to popular recommendations")
        try:
            recommendations = process_popular_recommendation()
            print(f"GET_REALTIME_RECOMMENDATIONS - Using {len(recommendations)} popular recommendations as fallback")
        except Exception as e:
            print(f"GET_REALTIME_RECOMMENDATIONS - Error getting popular recommendations: {str(e)}")
    
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

@app.route('/admin/monitor', methods=['POST'])
def manage_mongodb_monitor():
    """
    API endpoint để quản lý MongoDB monitor
    
    Query parameters:
    - action: Hành động quản lý (start, stop, status, force_update)
    """
    global mongodb_monitor
    
    if mongodb_monitor is None:
        # Khởi tạo MongoDB monitor nếu chưa tồn tại
        mongodb_monitor = MongoDBChangeMonitor(update_callback=reinitialize_models, debounce_seconds=300)
    
    action = request.args.get('action', default='status', type=str)
    
    if action == 'start':
        # Bắt đầu giám sát MongoDB
        result = mongodb_monitor.start()
        return jsonify({"status": "started" if result else "already_running"})
    
    elif action == 'stop':
        # Dừng giám sát MongoDB
        result = mongodb_monitor.stop()
        return jsonify({"status": "stopped" if result else "not_running"})
    
    elif action == 'status':
        # Kiểm tra trạng thái của MongoDB monitor
        is_running = mongodb_monitor.monitor_thread is not None and mongodb_monitor.monitor_thread.is_alive()
        return jsonify({"status": "running" if is_running else "stopped"})
    
    elif action == 'force_update':
        # Cưỡng chế cập nhật mô hình ngay lập tức
        result = mongodb_monitor.force_update()
        return jsonify({"status": "updated" if result else "failed"})
    
    else:
        return jsonify({"error": f"Invalid action: {action}"}), 400

@app.route('/demo/realtime')
def realtime_demo():
    """
    Serve the real-time recommendation demo page
    """
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'realtime_demo.html'), 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/admin/import', methods=['POST'])
def import_data():
    """
    API endpoint để nhập dữ liệu từ file và cập nhật mô hình đề xuất
    
    Request body:
    - file: File dữ liệu (CSV hoặc JSON)
    - collection: Tên collection trong MongoDB (mặc định: places)
    """
    global data_importer
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    collection = request.form.get('collection', default='places', type=str)
    
    # Tạo thư mục tạm để lưu file
    import tempfile
    from werkzeug.utils import secure_filename
    
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)
        
        # Gọi phương thức nhập dữ liệu tương ứng với định dạng file
        if file.filename.lower().endswith('.csv'):
            result = data_importer.import_csv(file_path, collection)
        elif file.filename.lower().endswith('.json'):
            result = data_importer.import_json(file_path, collection)
        else:
            return jsonify({"error": "Unsupported file format. Please use CSV or JSON"}), 400
    
    return jsonify(result)

@app.route('/webhook/data-update', methods=['POST'])
def data_update_webhook():
    """
    Webhook nhận thông báo khi có dữ liệu mới để cập nhật hệ thống đề xuất
    
    Request body:
    - type: Loại cập nhật (ví dụ: 'location', 'rating')
    - data_url: URL để tải dữ liệu mới (tuỳ chọn)
    """
    try:
        data = request.get_json()
        update_type = data.get('type', 'unknown')
        data_url = data.get('data_url')
        
        # Ghi log thông tin cập nhật
        app.logger.info(f"Received data update webhook: type={update_type}, url={data_url}")
        
        # Cưỡng chế cập nhật mô hình nếu có MongoDB monitor
        if mongodb_monitor:
            mongodb_monitor.force_update()
            return jsonify({"status": "update_triggered", "type": update_type})
        else:
            return jsonify({"status": "monitor_not_initialized", "type": update_type}), 500
    
    except Exception as e:
        app.logger.error(f"Error processing data update webhook: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)