from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
from scripts.preprocess import preprocess_data
from scripts.train_model import train_all_models
from scripts.generate_recommendations import generate_recommendations
from recommenders.hybrid import hybrid_recommendations
from recommenders.event_based import recommend_based_on_events
from services.chat_service import ChatService

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
from recommenders.realtime import RealtimeRecommender

# Khởi tạo RealtimeRecommender
realtime_recommender = RealtimeRecommender()

# Khởi tạo RealtimeIntegration
realtime_integration = None  # Will be initialized after data is loaded

# Khởi tạo services
chat_service = ChatService()


# Đường dẫn đến dữ liệu
RATINGS_PATH = "data/raw/dataset.csv"
PRODUCTS_PATH = "data/raw/products.csv"
EVENTS_PATH = "data/raw/events.csv"



print("Processing data...")
data = preprocess_data(RATINGS_PATH, PRODUCTS_PATH, EVENTS_PATH)
print(data.head()) 

# Xử lý dữ liệu
print("Processing data...")
data = preprocess_data(RATINGS_PATH, PRODUCTS_PATH, EVENTS_PATH)
product_details = pd.read_csv(PRODUCTS_PATH)

# Huấn luyện mô hình
print("Training models...")
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
        user_id = 1  # ID người dùng cần gợi ý
        recommendations = hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details)
        recommendations = recommendations.to_dict(orient='records')
        print(recommendations)
        # recommendations = generate_recommendations( collaborative_model, content_similarity,user_id, product_details)
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
        recommendations = content_similarity[product_id].tolist()
    else:
        return jsonify({"error": "Invalid case or missing product_id for content-based recommendations"}), 400

    return jsonify({"user_id": user_id, "recommendations": recommendations})

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
        recommendations = hybrid_recommendations(
            collaborative_model=collaborative_model,
            content_model_data=content_similarity,
            user_id=user_id,
            product_details=product_details,
            top_n=5
        )
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations.to_dict(orient='records') if isinstance(recommendations, pd.DataFrame) else recommendations
        })
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

# Điểm khởi chạy ứng dụng
if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)