"""
Utility class to manage the integration between real-time recommendations and WebSocket communications
"""
import logging
from flask_socketio import emit
from datetime import datetime
import pandas as pd
import json

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
            Type of event (e.g. 'view', 'add_to_cart', 'purchase')
        timestamp : datetime, optional
            Timestamp of the event, defaults to current time
        emit_recommendations : bool, optional
            Whether to emit recommendations to the user after recording the event
            
        Returns:
        --------
        tuple
            (success, recommendations, emitted)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            # Record the event in the recommender
            self.recommender.record_event(user_id, product_id, event_type, timestamp)
            
            if emit_recommendations:
                # Get recommendations based on the event
                recommendations = self.recommender.get_realtime_recommendations(
                    user_id=user_id,
                    current_product_id=product_id,
                    event_type=event_type,
                    product_details=self.product_details,
                    collaborative_model=self.collaborative_model,
                    content_data=self.content_data
                )
                
                # Store recent recommendations
                if user_id not in self.recent_recommendations:
                    self.recent_recommendations[user_id] = []
                    
                self.recent_recommendations[user_id].append({
                    'timestamp': datetime.now(),
                    'recommendations': recommendations
                })
                
                # Limit the number of stored recommendations
                if len(self.recent_recommendations[user_id]) > self.max_recent_recommendations:
                    self.recent_recommendations[user_id] = self.recent_recommendations[user_id][-self.max_recent_recommendations:]
                
                # Emit recommendations if user is connected
                emitted = self.send_recommendations(user_id, recommendations)
                
                return True, recommendations, emitted
            
            return True, None, False
                
        except Exception as e:
            logger.error(f"Error recording event: {str(e)}")
            return False, None, False
    
    def send_recommendations(self, user_id, recommendations):
        """
        Send recommendations to a user via their socket connection
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        recommendations : list
            List of recommendations to send
            
        Returns:
        --------
        bool
            Whether recommendations were sent
        """
        if user_id in self.user_socket_connections:
            socket_id = self.user_socket_connections[user_id]
            try:
                # Emit the recommendations to the user
                self.socketio.emit(
                    'realtime_recommendation', 
                    {'user_id': user_id, 'recommendations': recommendations},
                    room=socket_id
                )
                logger.info(f"Sent {len(recommendations)} recommendations to user {user_id}")
                return True
            except Exception as e:
                logger.error(f"Error sending recommendations to user {user_id}: {str(e)}")
        else:
            logger.info(f"User {user_id} not connected, recommendations not sent")
        
        return False
    
    def get_recent_recommendations(self, user_id, limit=10):
        """
        Get recent recommendations for a user
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
        limit : int, optional
            Maximum number of recent recommendations to return
            
        Returns:
        --------
        list
            Recent recommendations for the user
        """
        if user_id in self.recent_recommendations:
            sorted_recs = sorted(
                self.recent_recommendations[user_id],
                key=lambda r: r['timestamp'],
                reverse=True
            )
            return sorted_recs[:limit]
        
        return []
    
    def is_user_connected(self, user_id):
        """
        Check if a user is currently connected
        
        Parameters:
        -----------
        user_id : int or str
            ID of the user
            
        Returns:
        --------
        bool
            Whether the user is connected
        """
        return user_id in self.user_socket_connections

    def update_recommendations_for_all_users(self):
        """
        Update recommendations for all connected users
        
        Returns:
        --------
        dict
            Dictionary with user_ids as keys and boolean values indicating if recommendations were sent
        """
        results = {}
        
        # Only process for connected users
        for user_id in self.user_socket_connections.keys():
            try:
                # Get fresh recommendations for this user
                recommendations = self.recommender.get_realtime_recommendations(
                    user_id=user_id,
                    product_details=self.product_details,
                    collaborative_model=self.collaborative_model,
                    content_data=self.content_data,
                    top_n=10
                )
                
                # Send the recommendations
                sent = self.send_recommendations(user_id, recommendations)
                results[user_id] = sent
                
                # Store in recent recommendations if sent
                if sent:
                    if user_id not in self.recent_recommendations:
                        self.recent_recommendations[user_id] = []
                    
                    self.recent_recommendations[user_id].append({
                        'timestamp': datetime.now(),
                        'recommendations': recommendations
                    })
                    
                    # Limit the number of stored recommendations
                    if len(self.recent_recommendations[user_id]) > self.max_recent_recommendations:
                        self.recent_recommendations[user_id] = self.recent_recommendations[user_id][-self.max_recent_recommendations:]
                
            except Exception as e:
                logger.error(f"Error updating recommendations for user {user_id}: {str(e)}")
                results[user_id] = False
                
        return results
