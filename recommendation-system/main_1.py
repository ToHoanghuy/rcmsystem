from flask import Flask, request, jsonify, send_file, redirect
import pandas as pd
import os
import logging
from datetime import datetime
from scripts.preprocess import preprocess_data
from scripts.train_model import train_all_models
from scripts.generate_recommendations import generate_recommendations
from recommenders.hybrid import hybrid_recommendations
from recommenders.event_based import recommend_based_on_events
from recommenders.content_based import get_similar_products
from services.chat_service import ChatService
from utils.mongodb_loader import load_ratings_from_mongodb, load_events_from_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# Khởi tạo CORS để cho phép cross-origin requests
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})
# Khởi tạo SocketIO
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo utils cho realtime
from utils.realtime_integration import RealtimeIntegration
from recommenders.realtime_fixed import RealtimeRecommender

# Khởi tạo RealtimeRecommender
realtime_recommender = RealtimeRecommender()

# Khởi tạo RealtimeIntegration
realtime_integration = None  # Will be initialized after data is loaded

# Khởi tạo services
chat_service = ChatService()


# Đường dẫn đến dữ liệu - sử dụng dữ liệu địa điểm thay vì sản phẩm
RATINGS_PATH = "data/raw/location_ratings.csv"
PRODUCTS_PATH = "data/raw/locations.csv"  # Giữ tên biến để tương thích với code khác
EVENTS_PATH = "data/raw/location_events.csv"

print("Processing data...")
try:
    # Load ratings from MongoDB Review collection
    ratings_df = load_ratings_from_mongodb()
    logger.info(f"Loaded {len(ratings_df)} ratings from MongoDB Review collection.")
    logger.info(f"Ratings columns: {ratings_df.columns.tolist()}")
    if not ratings_df.empty:
        logger.info(f"Sample ratings data:\n{ratings_df.head(3)}")
    else:
        logger.warning("No ratings found in MongoDB Review collection.")

    # Check required columns
    required_cols = {"user_id", "location_id", "rating", "timestamp"}
    if ratings_df.empty or not required_cols.issubset(set(ratings_df.columns)):
        logger.error(f"Ratings data missing or malformed. Required columns: {required_cols}. Found: {ratings_df.columns.tolist()}")
        raise RuntimeError("No suitable rating data found in MongoDB. Please ensure the Review collection is populated with user_id, location_id, rating, timestamp.")

    # Load product/location details (still from CSV or MongoDB as appropriate)
    from utils.mongodb_loader import load_mongodb_locations
    product_details = load_mongodb_locations()
    if product_details.empty:
        logger.error("No location data found in MongoDB. Please ensure the locations collection is populated.")
        raise RuntimeError("No location data found in MongoDB.")
    logger.info(f"Loaded {len(product_details)} locations from MongoDB.")

    # Load events from JSON
    events_df = load_events_from_json()
    logger.info(f"Loaded {len(events_df)} events from JSON.")
    if not events_df.empty:
        logger.info(f"Sample events data:\n{events_df.head(3)}")
    else:
        logger.warning("No events found in location_events_full.json.")

    # Preprocess data using ratings and events
    from scripts.preprocess import preprocess_data
    data = preprocess_data(ratings_df=ratings_df, products_df=product_details, events_df=events_df)
    logger.info(f"Processed data: {len(data)} records")

except Exception as e:
    logger.error(f"Error processing data: {str(e)}")
    raise SystemExit("Aborting: Unable to load required data from MongoDB. See logs above.")

# Huấn luyện mô hình
print("Training models...")
# Kiểm tra xem đang sử dụng dữ liệu location hay product
if PRODUCTS_PATH.endswith('locations.csv'):
    print("Sử dụng mô hình location...")
    from scripts.train_model import train_all_location_models
    collaborative_model, content_similarity = train_all_location_models(data, product_details)
else:
    print("Sử dụng mô hình product...")
    collaborative_model, content_similarity = train_all_models(data, product_details)

# Initialize realtime integration with models
realtime_integration = RealtimeIntegration(
    socketio, 
    realtime_recommender,
    product_details=product_details,
    collaborative_model=collaborative_model,
    content_data=content_similarity
)

# API gợi ý
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    case = request.args.get('case', default='hybrid', type=str)
    product_id = request.args.get('product_id', type=int)

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    if case == 'behavior_based':
        # Gợi ý dựa trên sự kiện người dùng
        recommendations = recommend_based_on_events(user_id, EVENTS_PATH, product_details)
        recommendations = recommendations.to_dict(orient='records')
    elif case == 'hybrid':
        # Handle MongoDB-style IDs (hex strings) by keeping them as strings
        # Only try to convert to int if it's a numeric string for backward compatibility
        try:
            if isinstance(user_id, str) and user_id.isdigit():
                user_id = int(user_id)
        except (ValueError, TypeError):
            pass  # Keep the ID as is if it's not a numeric string
            
        print(f"Generating hybrid recommendations for user {user_id}...")
        recommendations = hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details)
        
        # Thêm thông tin bổ sung nếu đây là dữ liệu địa điểm
        if 'province' in product_details.columns:
            # Đây là dữ liệu địa điểm, bổ sung thông tin
            recommendations = recommendations.to_dict(orient='records')
            print(f"Generated {len(recommendations)} recommendations for user {user_id}")
        else:
            # Đây là dữ liệu sản phẩm thông thường
            recommendations = recommendations.to_dict(orient='records')
            print(f"Generated {len(recommendations)} product recommendations for user {user_id}")
    elif case == 'collaborative':
        if product_id is None:
            # Dự đoán cho tất cả sản phẩm
            product_ids = product_details['product_id'].tolist()
            recommendations = []
            
            for product_id in product_ids:
                try:
                    try:
                        prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                    except (ValueError, TypeError) as e:
                        # Nếu lỗi kiểu dữ liệu, thử với dạng chuỗi
                        print(f"Converting IDs to string in main.py: {e}")
                        prediction = collaborative_model.predict(uid=str(user_id), iid=str(product_id))
                    recommendations.append({
                        "product_id": product_id,
                        "predicted_rating": prediction.est
                    })
                except Exception as e:
                    print(f"Prediction failed for user {user_id}, product {product_id}: {e}")
                    # Nếu không dự đoán được, sử dụng giá trị mặc định
                    recommendations.append({
                        "product_id": product_id,
                        "predicted_rating": 3.0
                    })
        else:
            # Dự đoán cho một sản phẩm cụ thể
            try:
                try:
                    prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                except (ValueError, TypeError) as e:
                    # Nếu lỗi kiểu dữ liệu, thử với dạng chuỗi
                    print(f"Converting single IDs to string in main.py: {e}")
                    prediction = collaborative_model.predict(uid=str(user_id), iid=str(product_id))
                recommendations = {"product_id": product_id, "predicted_rating": prediction.est}
            except Exception as e:
                print(f"Single prediction failed for user {user_id}, product {product_id}: {e}")
                # Nếu không dự đoán được, sử dụng giá trị mặc định
                recommendations = {"product_id": product_id, "predicted_rating": 3.0}
    elif case == 'content_based' and product_id is not None:
        recommendations = content_similarity[product_id].tolist()
    else:
        return jsonify({"error": "Invalid case or missing product_id for content-based recommendations"}), 400

    return jsonify({"user_id": user_id, "recommendations": recommendations})

@app.route('/recommend_legacy/', methods=['GET'])
def recommend_legacy():
    """
    Legacy API endpoint for backward compatibility
    Redirects to the main recommend endpoint
    """
    try:
        # Get all parameters from the request
        user_id = request.args.get('user_id')
        case = request.args.get('case', default='hybrid', type=str)
        product_id = request.args.get('product_id')
        method = request.args.get('method', default='adaptive', type=str)
        
        # Log the request for debugging
        logger.info(f"Legacy recommendation request: user_id={user_id}, case={case}, product_id={product_id}")
        
        # Parameter validation
        if user_id is None and case not in ['content_based', 'popular']:
            return jsonify({"error": "user_id is required"}), 400
        
        if case == 'content_based' and product_id is None:
            return jsonify({"error": "product_id is required for content-based recommendations"}), 400
        
        # Serve recommendation directly
        if case == 'behavior_based':
            recommendations = recommend_based_on_events(user_id, EVENTS_PATH, product_details)
            recommendations = recommendations.to_dict(orient='records')
        elif case == 'hybrid':
            recommendations = hybrid_recommendations(
                collaborative_model=collaborative_model,
                content_model_data=content_similarity,
                user_id=user_id,
                product_details=product_details,
                method=method
            )
            recommendations = recommendations.to_dict(orient='records') if isinstance(recommendations, pd.DataFrame) else recommendations
        elif case == 'collaborative':
            if product_id is None:
                # Predict for all products
                product_ids = product_details['product_id'].tolist()
                recommendations = []
                for pid in product_ids:
                    try:
                        prediction = collaborative_model.predict(uid=user_id, iid=pid)
                        recommendations.append({
                            "product_id": pid,
                            "predicted_rating": prediction.est
                        })
                    except Exception as e:
                        print(f"Prediction failed for user {user_id}, product {pid}: {e}")
            else:
                # Predict for a specific product
                try:
                    prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                    recommendations = {"product_id": product_id, "predicted_rating": prediction.est}
                except Exception as e:
                    print(f"Single prediction failed for user {user_id}, product {product_id}: {e}")
                    recommendations = {"product_id": product_id, "predicted_rating": 3.0}
        elif case == 'content_based' and product_id is not None:
            # Handle content-based recommendations
            if isinstance(content_similarity, tuple):
                similarity_matrix, index_to_id, id_to_index = content_similarity
                if product_id in id_to_index:
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
                    # Product ID not found in similarity matrix
                    recommendations = []
            else:
                # Old format compatibility
                recommendations = content_similarity[product_id].tolist() if product_id in content_similarity else []
        else:
            return jsonify({"error": "Invalid case or missing product_id for content-based recommendations"}), 400
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "from_cache": False,
            "case": case
        })
        
    except Exception as e:
        logger.error(f"Error in recommend_legacy: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
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
    data = request.get_json()
    user_id = data.get('user_id')
    product_id = data.get('product_id')
    event_type = data.get('event_type')
    
    if not all([user_id, product_id, event_type]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Record the event and get recommendations
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate recommendations based on event
        recommendations = hybrid_recommendations(
            collaborative_model=collaborative_model,
            content_model_data=content_similarity,
            user_id=user_id,
            product_details=product_details,
            top_n=5
        )
        
        # Signal to WebSocket client that we're sending via socket
        sent_via_socket = False
        
        return jsonify({
            "status": "success", 
            "event_recorded": True,
            "recommendations": recommendations.to_dict(orient='records') if isinstance(recommendations, pd.DataFrame) else recommendations,
            "sent_via_socket": sent_via_socket
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/realtime-recommend', methods=['GET'])
def realtime_recommend():
    """
    Get real-time recommendations for a user
    """
    user_id = request.args.get('user_id', type=int)
    product_id = request.args.get('product_id', type=int)
    
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
    
    try:
        # Get recommendations
        if realtime_integration:
            recommendations = realtime_integration.get_recommendations(
                user_id=user_id,
                context_item=product_id,
                limit=10
            )
        else:
            recommendations = hybrid_recommendations(
                collaborative_model=collaborative_model,
                content_model_data=content_similarity,
                user_id=user_id,
                product_details=product_details,
                top_n=5
            )
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/websocket-test')
def websocket_test():
    """
    Test client for WebSocket integration
    """
    try:
        return send_file('static/websocket_test.html')
    except Exception as e:
        return f"Error: {str(e)}", 500

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
    from flask import redirect
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

# Initialize realtime_integration after loading models
realtime_integration = RealtimeIntegration(socketio, realtime_recommender, product_details)

# Import and register API endpoints
from services.api_endpoints import register_api_endpoints
from utils.optimized_cache import OptimizedRecommendationCache
from utils.mongodb_store import MongoDBEventStore

# Create an optimized recommendation cache
recommendation_cache = OptimizedRecommendationCache(expiry_seconds=3600, max_size=1000)

# Initialize MongoDB store if MongoDB connection is available
mongodb_store = None
try:
    mongodb_store = MongoDBEventStore()
    logger.info("MongoDB event store initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize MongoDB event store: {str(e)}")
    logger.warning("Events will only be stored in local files")

# Register API endpoints
register_api_endpoints(
    app=app, 
    realtime_recommender=realtime_recommender, 
    realtime_integration=realtime_integration,
    recommendation_cache=recommendation_cache,
    event_store=mongodb_store,
    DATA_DIR=os.path.dirname(RATINGS_PATH)
)

# Điểm khởi chạy ứng dụng
if __name__ == "__main__":
    # Xác định loại dữ liệu đang sử dụng
    data_type = "địa điểm" if PRODUCTS_PATH.endswith('locations.csv') else "sản phẩm"
    print(f"\nKhởi động API server đề xuất với dữ liệu {data_type}...")
    print(f"Nguồn dữ liệu: {RATINGS_PATH}, {PRODUCTS_PATH}")
    print(f"Server đang chạy tại http://127.0.0.1:5000/")
    print(f"Thử: http://127.0.0.1:5000/recommend?user_id=1&case=hybrid")
    print(f"Demo: http://127.0.0.1:5000/demo/realtime\n")
    socketio.run(app, debug=True, port=5000)