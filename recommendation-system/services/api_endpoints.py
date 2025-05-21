"""
API endpoints để tích hợp Frontend với hệ thống đề xuất
"""

from flask import Flask, request, jsonify
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def register_api_endpoints(app, realtime_recommender, realtime_integration, recommendation_cache, event_store=None, DATA_DIR=None, skip_endpoints=None):
    """
    Đăng ký các API endpoints cho hệ thống đề xuất
    
    Parameters:
    -----------
    app: Flask
        Ứng dụng Flask
    realtime_recommender: RealtimeRecommender
        Đối tượng RealtimeRecommender để xử lý sự kiện
    realtime_integration: RealtimeIntegration  
        Đối tượng RealtimeIntegration để xử lý WebSocket
    recommendation_cache: RecommendationCache
        Đối tượng cache cho các đề xuất
    event_store: MongoDBEventStore, optional
        Đối tượng lưu trữ sự kiện vào MongoDB
    DATA_DIR: str, optional
        Đường dẫn đến thư mục data
    skip_endpoints: list, optional
        Danh sách các endpoints cần bỏ qua (đã được đăng ký trước đó)
    """
    
    if skip_endpoints is None:
        skip_endpoints = []
        
    if DATA_DIR is None:
        # Sử dụng đường dẫn mặc định
        DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    @app.route('/api/track', methods=['POST'])
    def track_event():
        """
        API endpoint để thu thập sự kiện người dùng từ frontend
        
        Body JSON:
        {
            "user_id": "id_user",
            "event_type": "view" | "click" | "book" | "rate" | "search" | "favorite",
            "location_id": "id_location",
            "timestamp": "2025-05-19T15:30:45Z", (tùy chọn, mặc định là thời gian hiện tại)
            "data": {
                "rating": 4.5,           (cho sự kiện rate)
                "session_id": "abc123",   (tùy chọn)
                "search_query": "string", (cho sự kiện search)
                "device_info": {},        (tùy chọn)
                ...
            }
        }
        """
        try:
            # Lấy dữ liệu sự kiện từ request
            event_data = request.json
            
            if not event_data or not isinstance(event_data, dict):
                return jsonify({"success": False, "error": "Invalid event data format"}), 400
                
            # Kiểm tra các trường bắt buộc
            required_fields = ["user_id", "event_type", "location_id"]
            for field in required_fields:
                if field not in event_data:
                    return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
            
            # Thêm timestamp nếu không có
            if "timestamp" not in event_data:
                event_data["timestamp"] = datetime.now().isoformat()
                
            # Đảm bảo trường data tồn tại
            if "data" not in event_data:
                event_data["data"] = {}
                
            # Xử lý sự kiện theo loại
            event_type = event_data["event_type"]
            user_id = event_data["user_id"]
            location_id = event_data["location_id"]            # Lưu sự kiện vào bộ nhớ của RealtimeRecommender
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
                    
                    # Nếu là sự kiện rating, cập nhật dữ liệu ratings
                    if event_type == "rate" and "rating" in event_data["data"]:
                        rating_value = float(event_data["data"]["rating"])
                        realtime_recommender.add_rating(user_id, location_id, rating_value)
                except Exception as e:
                    logger.error(f"Error processing event: {str(e)}")
                    # We'll continue processing despite the error to save event to CSV
                
                # Lưu sự kiện vào MongoDB nếu có
                if event_store:
                    event_store.store_event(event_data)
                
                # Lưu sự kiện vào file CSV để sử dụng sau này
                save_event_to_csv(event_data, DATA_DIR)                # Cập nhật mô hình real-time và gửi đề xuất
                recommendations = None
                if realtime_integration:
                    try:
                        # Ensure proper timestamp parsing with error handling
                        try:
                            timestamp = datetime.fromisoformat(event_data["timestamp"].replace('Z', '+00:00'))
                        except (ValueError, KeyError):
                            timestamp = datetime.now()
                            
                        # Fix: Change location_id to product_id to match the parameter name in realtime_integration
                        success, recs = realtime_integration.record_event(
                            user_id, 
                            product_id=location_id,  # Use named parameter to avoid confusion
                            event_type=event_type, 
                            timestamp=timestamp,
                            emit_recommendations=True
                        )
                        recommendations = recs
                    except Exception as e:
                        logger.error(f"Error in realtime integration: {str(e)}")
                        recommendations = None
                
                # Kiểm tra xem user đã đăng ký với Socket.IO chưa
                user_connected = False
                if realtime_integration and realtime_integration.user_socket_connections.get(user_id):
                    user_connected = True
                
                return jsonify({
                    "success": True, 
                    "message": "Event tracked successfully",
                    "sent_via_socket": user_connected,
                    "recommendations": recommendations if not user_connected else None
                }), 200
            else:
                return jsonify({"success": False, "error": "Realtime recommender not initialized"}), 500
                
        except Exception as e:
            logger.error(f"Error tracking event: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/api/recommend', methods=['GET'])
    def recommend():
        """
        API endpoint cho việc gợi ý sản phẩm
        
        Query parameters:
        - user_id: ID của người dùng
        - case: Loại gợi ý (collaborative, content_based, hybrid, event_based)
        - location_id: ID của địa điểm (chỉ cần cho content_based)
        - method: Phương pháp hybrid (adaptive, switching)
        """
        try:
            # Lấy các tham số từ request
            user_id = request.args.get('user_id')
            case = request.args.get('case', default='hybrid', type=str)
            location_id = request.args.get('location_id')
            method = request.args.get('method', default='adaptive', type=str)
            
            # Kiểm tra tham số
            if user_id is None and case not in ['content_based', 'popular']:
                return jsonify({"success": False, "error": "user_id is required"}), 400
            
            if case == 'content_based' and location_id is None:
                return jsonify({"success": False, "error": "location_id is required for content_based recommendations"}), 400
                
            # Kiểm tra cache
            cache_key = f"{user_id}_{case}_{location_id}_{method}"
            cached_result = recommendation_cache.get(cache_key)
            if cached_result:
                return jsonify({"success": True, "recommendations": cached_result}), 200
                
            # TODO: Implement recommendation logic
            # Trả về dummy recommendations tạm thời
            recommendations = [
                {"location_id": "123", "name": "Example Location 1", "score": 0.9},
                {"location_id": "456", "name": "Example Location 2", "score": 0.8},
            ]
            
            # Lưu vào cache
            recommendation_cache.set(cache_key, recommendations)
            
            return jsonify({"success": True, "recommendations": recommendations}), 200
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500

def save_event_to_csv(event_data, data_dir):
    """
    Lưu sự kiện vào file CSV
    
    Parameters:
    -----------
    event_data: dict
        Dữ liệu sự kiện
    data_dir: str
        Đường dẫn đến thư mục data
    """
    import pandas as pd
    import os
    import json
    
    try:
        # Đường dẫn đến file sự kiện
        events_file = os.path.join(data_dir, 'raw', 'location_events.csv')
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(events_file), exist_ok=True)
        
        # Prepare event data for CSV storage
        # Create a simplified version that follows the same format as existing data
        simple_event = {
            'user_id': str(event_data.get('user_id')),
            'product_id': str(event_data.get('location_id')),
            'event_type': event_data.get('event_type'),
            'timestamp': event_data.get('timestamp')
        }
          # Store the full data separately in a new file to ensure compatibility
        full_events_file = os.path.join(data_dir, 'raw', 'location_events_full.json')
        
        with open(full_events_file, 'a') as f:
            f.write(json.dumps(event_data) + '\n')
            
        # Create DataFrame from simplified event
        event_df = pd.DataFrame([simple_event])
        
        # Check if file exists
        file_exists = os.path.isfile(events_file)
        
        # Write to CSV, append if file exists
        event_df.to_csv(events_file, mode='a', header=not file_exists, index=False)
        
        logger.info(f"Event saved to CSV: {simple_event}")
        return True
    except Exception as e:
        logger.error(f"Error saving event to CSV: {str(e)}")
        return False
