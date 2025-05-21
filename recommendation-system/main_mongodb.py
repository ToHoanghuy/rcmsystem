"""
Modified version of main.py that uses MongoDB data instead of CSV files.
Run this file to start the recommendation system using MongoDB data.
"""

from flask import Flask, request, jsonify, send_file, redirect
import pandas as pd
import os
import logging
from datetime import datetime
from scripts.preprocess import preprocess_pipeline
from scripts.train_model import train_all_models
from scripts.generate_recommendations import generate_recommendations
from recommenders.hybrid import hybrid_recommendations
from recommenders.event_based import recommend_based_on_events
from services.chat_service import ChatService
from database.mongodb import MongoDB
from utils.mongodb_loader import load_data_from_mongodb_or_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Initialize CORS to allow cross-origin requests
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})
# Initialize SocketIO
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize utils for realtime
from utils.realtime_integration import RealtimeIntegration
from recommenders.realtime_fixed import RealtimeRecommender

# Initialize RealtimeRecommender
realtime_recommender = RealtimeRecommender()

# Initialize RealtimeIntegration
realtime_integration = None  # Will be initialized after data is loaded

# Initialize services
chat_service = ChatService()

# Initialize MongoDB connection
mongodb = None
try:
    mongodb = MongoDB()
    logger.info("MongoDB connection initialized successfully")
except Exception as e:
    logger.error(f"Error initializing MongoDB connection: {str(e)}")
    logger.warning("Falling back to CSV data")
    mongodb = None

print("Loading and processing data...")
try:
    # Try to load data from MongoDB first
    product_details = load_data_from_mongodb_or_csv('locations')
    
    if product_details.empty:
        logger.warning("No location data found in MongoDB. Falling back to CSV files.")
        # Default paths for CSV files as fallback
        RATINGS_PATH = "data/raw/location_ratings.csv"
        PRODUCTS_PATH = "data/raw/locations.csv"
        EVENTS_PATH = "data/raw/location_events.csv"
        
        # Check if files exist
        if not os.path.exists(RATINGS_PATH):
            logger.warning(f"File not found: {RATINGS_PATH}. Using dataset.csv as fallback.")
            RATINGS_PATH = "data/raw/dataset.csv"
        
        if not os.path.exists(PRODUCTS_PATH):
            logger.warning(f"File not found: {PRODUCTS_PATH}. Using products.csv as fallback.")
            PRODUCTS_PATH = "data/raw/products.csv"
        
        if not os.path.exists(EVENTS_PATH):
            logger.warning(f"File not found: {EVENTS_PATH}. Using events.csv as fallback.")
            EVENTS_PATH = "data/raw/events.csv" if os.path.exists("data/raw/events.csv") else None
        
        # Preprocess data from CSV files
        data = preprocess_pipeline(use_mongodb=False)
        product_details = pd.read_csv(PRODUCTS_PATH)
    else:
        # Process data from MongoDB
        logger.info(f"Loaded {len(product_details)} locations from MongoDB")
        
        # Load events and ratings data from MongoDB
        events_data = load_data_from_mongodb_or_csv('events')
        ratings_data = load_data_from_mongodb_or_csv('ratings')
        
        # Preprocess data using our custom pipeline
        data = preprocess_pipeline(
            ratings_df=ratings_data,
            products_df=product_details,
            events_df=events_data,
            use_mongodb=False  # We already loaded the data from MongoDB
        )
        
        logger.info(f"Processed data from MongoDB: {len(data)} records")
    
except Exception as e:
    logger.error(f"Error processing data: {str(e)}")
    logger.warning("Falling back to default product data")
    RATINGS_PATH = "data/raw/dataset.csv"
    PRODUCTS_PATH = "data/raw/products.csv"
    EVENTS_PATH = "data/raw/events.csv"
    data = preprocess_pipeline(use_mongodb=False)
    product_details = pd.read_csv(PRODUCTS_PATH)

# Train models
print("Training models...")
# Check if we're using location or product data
if 'province' in product_details.columns:
    print("Using location models...")
    from scripts.train_model import train_all_location_models
    collaborative_model, content_similarity = train_all_location_models(data, product_details)
else:
    print("Using product models...")
    collaborative_model, content_similarity = train_all_models(data, product_details)

# Initialize realtime integration with models
realtime_integration = RealtimeIntegration(
    socketio, 
    realtime_recommender,
    product_details=product_details,
    collaborative_model=collaborative_model,
    content_data=content_similarity
)

# Initialize MongoDB event store
from utils.mongodb_store import MongoDBEventStore
mongodb_store = None
try:
    mongodb_store = MongoDBEventStore()
    logger.info("MongoDB event store initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize MongoDB event store: {str(e)}")
    logger.warning("Events will only be stored in local files")

# API endpoints
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    case = request.args.get('case', default='hybrid', type=str)
    product_id = request.args.get('product_id', type=int)

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
    if case == 'behavior_based':
        # Get recent events from MongoDB
        if mongodb_store:
            user_events = mongodb_store.get_user_events(user_id, limit=50)
            
            # Create events DataFrame for recommendation
            if user_events:
                events_df = pd.DataFrame(user_events)
                # Make sure events_df has the right columns
                recommendations = recommend_based_on_events(user_id, events_df, product_details)
            else:
                logger.warning(f"No events found for user {user_id} in MongoDB")
                # Fallback to hybrid recommendations
                recommendations = hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details)
        else:
            # Fallback to CSV if MongoDB is not available
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
        
        # Add additional info if this is location data
        if 'province' in product_details.columns:
            # This is location data, add additional info
            recommendations = recommendations.to_dict(orient='records')
            print(f"Generated {len(recommendations)} recommendations for user {user_id}")
        else:
            # This is regular product data
            recommendations = recommendations.to_dict(orient='records')
            print(f"Generated {len(recommendations)} product recommendations for user {user_id}")
    elif case == 'collaborative':
        if product_id is None:
            # Predict for all products
            product_ids = product_details['product_id'].tolist()
            recommendations = []
            for product_id in product_ids:
                try:
                    try:
                        prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                    except (ValueError, TypeError) as e:
                        # If type error, try with string format
                        print(f"Converting IDs to string in main.py: {e}")
                        prediction = collaborative_model.predict(uid=str(user_id), iid=str(product_id))
                    recommendations.append({
                        "product_id": product_id,
                        "predicted_rating": prediction.est
                    })
                except Exception as e:
                    print(f"Prediction failed for user {user_id}, product {product_id}: {e}")
                    # If prediction fails, use default value
                    recommendations.append({
                        "product_id": product_id,
                        "predicted_rating": 3.0
                    })
        else:
            # Predict for a specific product
            try:
                try:
                    prediction = collaborative_model.predict(uid=user_id, iid=product_id)
                except (ValueError, TypeError) as e:
                    # If type error, try with string format
                    print(f"Converting single IDs to string in main.py: {e}")
                    prediction = collaborative_model.predict(uid=str(user_id), iid=str(product_id))
                recommendations = {"product_id": product_id, "predicted_rating": prediction.est}
            except Exception as e:
                print(f"Single prediction failed for user {user_id}, product {product_id}: {e}")
                # If prediction fails, use default value
                recommendations = {"product_id": product_id, "predicted_rating": 3.0}
    elif case == 'content_based' and product_id is not None:
        recommendations = content_similarity[product_id].tolist()
    else:
        return jsonify({"error": "Invalid case or missing product_id for content-based recommendations"}), 400

    return jsonify({"user_id": user_id, "recommendations": recommendations})

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
        
        # Store event in MongoDB
        event_data = {
            'user_id': user_id,
            'location_id': product_id,  # Use location_id for consistency with MongoDB schema
            'product_id': product_id,   # Keep product_id for backward compatibility
            'event_type': event_type,
            'timestamp': timestamp,
            'data': data.get('data', {})
        }
        
        # Store in MongoDB if available
        mongodb_event_saved = False
        if mongodb_store:
            mongodb_event_saved = mongodb_store.store_event(event_data)
            
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
            "mongodb_saved": mongodb_event_saved,
            "recommendations": recommendations.to_dict(orient='records') if isinstance(recommendations, pd.DataFrame) else recommendations,
            "sent_via_socket": sent_via_socket
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/track', methods=['POST'])
def track_event():
    """
    API endpoint to collect user events from frontend
    
    Body JSON:
    {
        "user_id": "id_user",
        "event_type": "view" | "click" | "book" | "rate" | "search" | "favorite",
        "location_id": "id_location",
        "timestamp": "2025-05-19T15:30:45Z", (optional, default is current time)
        "data": {
            "rating": 4.5,           (for rate events)
            "session_id": "abc123",   (optional)
            "search_query": "string", (for search events)
            "device_info": {},        (optional)
            ...
        }
    }
    """
    try:
        # Get event data from request
        event_data = request.json
        
        if not event_data or not isinstance(event_data, dict):
            return jsonify({"success": False, "error": "Invalid event data format"}), 400
            
        # Check required fields
        required_fields = ["user_id", "event_type", "location_id"]
        for field in required_fields:
            if field not in event_data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        # Add timestamp if not present
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now().isoformat()
            
        # Ensure data field exists
        if "data" not in event_data:
            event_data["data"] = {}
            
        # Process event by type
        event_type = event_data["event_type"]
        user_id = event_data["user_id"]
        location_id = event_data["location_id"]
        
        # Store in realtime recommender memory
        if realtime_recommender:
            try:
                # Check if realtime_recommender has necessary attributes
                if not hasattr(realtime_recommender, 'recent_events'):
                    logger.warning("Initializing recent_events in realtime_recommender")
                    realtime_recommender.recent_events = {}
                    
                # Add the event with error handling
                success = realtime_recommender.add_event(user_id, location_id, event_type, event_data["data"])
                if not success:
                    logger.warning(f"Failed to add event for user {user_id}, location {location_id}")
                
                # If it's a rating event, update ratings data
                if event_type == "rate" and "rating" in event_data["data"]:
                    rating_value = float(event_data["data"]["rating"])
                    realtime_recommender.add_rating(user_id, location_id, rating_value)
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
            
            # Store event in MongoDB if available
            if mongodb_store:
                mongodb_store.store_event(event_data)
            
            # Update realtime integration and send recommendations
            recommendations = None
            if realtime_integration:
                try:
                    # Ensure proper timestamp parsing with error handling
                    try:
                        timestamp = datetime.fromisoformat(event_data["timestamp"].replace('Z', '+00:00'))
                    except (ValueError, KeyError):
                        timestamp = datetime.now()
                        
                    # Use named parameter to avoid confusion
                    success, recs = realtime_integration.record_event(
                        user_id, 
                        product_id=location_id,
                        event_type=event_type, 
                        timestamp=timestamp,
                        emit_recommendations=True
                    )
                    recommendations = recs
                except Exception as e:
                    logger.error(f"Error in realtime integration: {str(e)}")
                    recommendations = None
            
            # Check if user is registered with Socket.IO
            user_connected = False
            if realtime_integration and realtime_integration.user_socket_connections.get(user_id):
                user_connected = True
            
            return jsonify({
                "success": True, 
                "message": "Event tracked successfully",
                "mongodb_saved": mongodb_store is not None,
                "sent_via_socket": user_connected,
                "recommendations": recommendations if not user_connected else None
            }), 200
        else:
            return jsonify({"success": False, "error": "Realtime recommender not initialized"}), 500
            
    except Exception as e:
        logger.error(f"Error tracking event: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/realtime-recommend', methods=['GET'])
def realtime_recommend():
    """
    Get real-time recommendations for a user
    """
    user_id = request.args.get('user_id')
    product_id = request.args.get('product_id')
    
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

# Initialize API endpoints
from services.api_endpoints import register_api_endpoints
from utils.optimized_cache import OptimizedRecommendationCache

# Create an optimized recommendation cache
recommendation_cache = OptimizedRecommendationCache(expiry_seconds=3600, max_size=1000)

# Register API endpoints - avoid registering /api/track since we already defined it
register_api_endpoints(
    app=app, 
    realtime_recommender=realtime_recommender, 
    realtime_integration=realtime_integration,
    recommendation_cache=recommendation_cache,
    event_store=mongodb_store,
    DATA_DIR="data/raw",
    skip_endpoints=['track_event']  # Skip the track_event endpoint that we've already defined
)

# Application entry point
if __name__ == "__main__":
    # Determine data type
    data_type = "location" if 'province' in product_details.columns else "product"
    data_source = "MongoDB" if mongodb else "CSV files"
    
    print(f"\nStarting recommendation API server with {data_type} data from {data_source}...")
    print(f"Server running at http://127.0.0.1:5000/")
    print(f"Try: http://127.0.0.1:5000/recommend?user_id=1&case=hybrid")
    print(f"Demo: http://127.0.0.1:5000/demo/realtime\n")
    
    socketio.run(app, debug=True, port=5000)
