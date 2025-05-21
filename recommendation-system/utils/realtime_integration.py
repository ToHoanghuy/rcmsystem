"""
Utility class to manage the integration between real-time recommendations and WebSocket communications
"""
import logging
from flask_socketio import emit
from datetime import datetime
import pandas as pd
import json
from utils.json_utils import convert_to_json_serializable, clean_json_data
from utils.mongodb_json_cleaner import clean_mongodb_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealtimeIntegration:
    """
    Class for integrating real-time recommendation events and WebSocket communications
    """
    def __init__(self, socketio, realtime_recommender, product_details=None, collaborative_model=None, content_data=None):
        # Store socket.io instance
        self.socketio = socketio
        # Store realtime recommender
        self.recommender = realtime_recommender
        # Store socket connections mapped to user_ids
        self.user_socket_connections = {}
        # Store recent recommendations
        self.recent_recommendations = {}
        # Maximum number of recent recommendations to store per user
        self.max_recent_recommendations = 10
        
        # Store model data for recommendations
        self.product_details = product_details
        self.collaborative_model = collaborative_model
        self.content_data = content_data
        
    def register_user(self, user_id, socket_id):
        """
        Register a user with a socket connection
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        socket_id : str
            Socket connection ID
        """
        self.user_socket_connections[user_id] = socket_id
        logger.info(f"Registered user {user_id} with socket ID {socket_id}")
        return True
        
    def unregister_user(self, user_id=None, socket_id=None):
        """
        Unregister a user from socket connections
        
        Parameters:
        -----------
        user_id : int or str, optional
            ID of the user to unregister
        socket_id : str, optional
            Socket connection ID to unregister
            
        Note: Either user_id or socket_id must be provided
        """
        if user_id is not None and user_id in self.user_socket_connections:
            del self.user_socket_connections[user_id]
            logger.info(f"Unregistered user {user_id}")
            return True
            
        elif socket_id is not None:
            users_to_remove = []
            for uid, sid in self.user_socket_connections.items():
                if sid == socket_id:
                    users_to_remove.append(uid)
            
            for uid in users_to_remove:
                del self.user_socket_connections[uid]
                logger.info(f"Unregistered user {uid} by socket ID {socket_id}")
            
            return len(users_to_remove) > 0
            
        return False
        
    def record_event(self, user_id, product_id, event_type, timestamp=None, emit_recommendations=True):
        """
        Record a user event and optionally emit recommendations
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        product_id : int or str
            ID of the product
        event_type : str
            Type of event (e.g. "view", "add_to_cart", "purchase")
        timestamp : datetime, optional
            Timestamp of the event, defaults to current time
        emit_recommendations : bool, optional
            Whether to emit recommendations to the user after recording the event
            
        Returns:
        --------
        tuple
            (success, recommendations)
        """
        # Handle IDs with a safer approach - don't force conversion to int
        # MongoDB IDs and other string IDs should be preserved as strings
        # Only convert to int if they're numeric strings for backward compatibility
        # Always ensure IDs are strings for consistent handling with MongoDB-style IDs
        user_id = str(user_id) if user_id is not None else None
        product_id = str(product_id) if product_id is not None else None
        logger.debug(f"Using string IDs: user_id={user_id}, product_id={product_id}")
        
        # Set timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Record the event in the recommender
        if self.recommender is not None:
            try:
                # Since realtime.py expects location_id, pass product_id as location_id
                success = self.recommender.record_event(user_id, product_id, event_type, timestamp)
            except Exception as e:
                logger.error(f"Error recording event in recommender: {str(e)}")
                success = False
        else:
            logger.warning("No recommender available to record event")
            success = False
            
        # Get recommendations
        recommendations = self.get_recommendations(user_id, product_id, limit=10)
        
        # Emit recommendations to the user if requested
        if emit_recommendations and user_id in self.user_socket_connections:
            self.send_recommendations(user_id, recommendations)
        
        return success, recommendations
        
    def get_recommendations(self, user_id, context_item=None, limit=10):
        """
        Get recommendations for a user
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        context_item : int or str, optional
            ID of the context item (e.g., current product)
        limit : int, optional
            Maximum number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended products
        """
        recommendations = []
        
        if self.recommender is not None:
            # Get recommendations from the recommender using get_realtime_recommendations method
            recommendations = self.recommender.get_realtime_recommendations(
                user_id=user_id,
                current_product_id=context_item,
                product_details=self.product_details,
                collaborative_model=self.collaborative_model,
                content_data=self.content_data,
                top_n=limit
            )            # Add product details if available
        if self.product_details is not None and len(recommendations) > 0:
            if isinstance(recommendations, pd.DataFrame):
                # If recommendations is already a DataFrame, merge product details
                # Convert all IDs to strings for safe comparison with MongoDB-style IDs
                recommendations['product_id'] = recommendations['product_id'].astype(str)
                
                # Determine ID column in product_details
                id_column = 'product_id' if 'product_id' in self.product_details.columns else 'location_id'
                
                # Create a copy of product_details with string IDs
                product_details_copy = self.product_details.copy()
                product_details_copy[id_column] = product_details_copy[id_column].astype(str)
                
                recommendations = pd.merge(
                    recommendations,
                    product_details_copy,
                    on="product_id",
                    how="left"
                )
            elif isinstance(recommendations, list) and all(isinstance(rec, dict) for rec in recommendations):
                # If recommendations is a list of dicts with product_id
                for rec in recommendations:
                    if 'product_id' in rec:
                        # Find matching product details
                        product_id = int(rec['product_id'])
                        product_info = self.product_details[self.product_details['product_id'] == product_id]
                        if not product_info.empty:
                            # Add all product details to recommendation
                            for col in product_info.columns:
                                if col != 'product_id' and col not in rec:
                                    rec[col] = product_info.iloc[0][col]
            else:
                # If recommendations is a list of product IDs, convert to DataFrame
                try:
                    rec_df = pd.DataFrame({"product_id": [int(pid) for pid in recommendations]})
                    recommendations = pd.merge(
                        rec_df,
                        self.product_details.astype({'product_id': int}),
                        on="product_id",
                        how="left"
                    ).to_dict('records')
                except (ValueError, TypeError):
                    # Skip merging if product IDs can't be converted to int
                    pass
                
        # Cache recommendations for this user
        self.cache_recommendations(user_id, recommendations)
                
        return recommendations
            
    def cache_recommendations(self, user_id, recommendations):
        """
        Cache recommendations for a user
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        recommendations : DataFrame or list
            Recommendations to cache
        """
        if user_id not in self.recent_recommendations:
            self.recent_recommendations[user_id] = []
            
        # Convert recommendations to a list if it's a DataFrame
        if isinstance(recommendations, pd.DataFrame):
            rec_list = recommendations.to_dict("records")
        else:
            rec_list = recommendations
            
        # Add current recommendations to the cache
        self.recent_recommendations[user_id].append({
            "timestamp": datetime.now(),
            "recommendations": rec_list
        })
        
        # Trim cache to max size
        while len(self.recent_recommendations[user_id]) > self.max_recent_recommendations:
            self.recent_recommendations[user_id].pop(0)
            
    def send_recommendations(self, user_id, recommendations):
        """
        Send recommendations to a user via Socket.IO
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        recommendations : DataFrame or list
            Recommendations to send
            
        Returns:
        --------
        bool
            Whether recommendations were sent successfully
        """
        if user_id not in self.user_socket_connections:
            logger.warning(f"User {user_id} not found in socket connections")
            return False
            
        socket_id = self.user_socket_connections[user_id]
        
        # Convert recommendations to a list if it's a DataFrame
        if isinstance(recommendations, pd.DataFrame):
            rec_list = recommendations.to_dict("records")
        else:
            rec_list = recommendations
            
        # Clean and convert any problematic data types for JSON serialization
        # This will handle NumPy types, MongoDB ObjectIDs, and Unicode characters properly
        # Sử dụng hàm nâng cao xử lý MongoDB JSON
        rec_list = clean_mongodb_json(rec_list)
            
        try:
            # Send recommendations via Socket.IO to the specific user
            self.socketio.emit(
                "realtime_recommendation",
                {
                    "user_id": user_id,
                    "recommendations": rec_list
                },
                room=socket_id
            )
            logger.info(f"Sent recommendations to user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending recommendations to user {user_id}: {str(e)}")
            return False

    def update_recommendations_for_all_users(self):
        """
        Update recommendations for all connected users
        
        Returns:
        --------
        dict
            Dictionary of user_id -> success status
        """
        results = {}
        
        for user_id in self.user_socket_connections:
            recommendations = self.get_recommendations(user_id, limit=10)
            success = self.send_recommendations(user_id, recommendations)
            results[user_id] = success
            
        return results
