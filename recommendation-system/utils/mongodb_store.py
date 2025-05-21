"""
MongoDB integration for storing recommendation events.
This module provides functionality to store user events in MongoDB
for later retrieval and model training.
"""

import os
import logging
import json
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBEventStore:
    """
    Store user events in MongoDB for later use in recommendation model training
    """
    def __init__(self, connection_string=None, db_name=None):
        """
        Initialize the MongoDB event store
        
        Parameters:
        -----------
        connection_string : str or None
            MongoDB connection string. If None, will try to use environment variable or Config.
        db_name : str or None
            Name of the database to use. If None, will use from Config or default.
        """
        # Get connection string from Config or environment variable if not provided
        if connection_string is None:
            connection_string = Config.MONGODB_URI
            
        if connection_string is None:
            connection_string = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
            
        # Get database name from Config or use default if not provided
        if db_name is None:
            db_name = os.environ.get('MONGODB_DATABASE', 'travel_recommendations')
        
        self.client = None
        self.db = None
        self.connection_string = connection_string
        self.db_name = db_name
        
        # Collections
        self.events_collection = None
        self.ratings_collection = None
        self.recommendations_collection = None
        self._connect()
    
    def _connect(self):
        """
        Connect to MongoDB server
        """
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            
            # Initialize collections
            self.events_collection = self.db['events']
            self.ratings_collection = self.db['ratings']
            self.recommendations_collection = self.db['recommendations']
            
            # Create indexes for faster queries
            self.events_collection.create_index([('user_id', 1)])
            self.events_collection.create_index([('location_id', 1)])
            self.events_collection.create_index([('event_type', 1)])
            self.events_collection.create_index([('timestamp', 1)])
            
            self.ratings_collection.create_index([('user_id', 1), ('location_id', 1)], unique=True)
            
            logger.info(f"Connected to MongoDB at {self.connection_string}")
            return True
        
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Could not connect to MongoDB: {str(e)}")
            # Set the db to None to indicate connection failure
            self.db = None
            self.events_collection = None
            self.ratings_collection = None
            self.recommendations_collection = None
            return False

    def store_event(self, event_data):
        """
        Store a user event in MongoDB
        
        Parameters:
        -----------
        event_data : dict
            Event data to store
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check if we have a valid db connection
        if self.db is None or self.events_collection is None:
            if not self._connect():
                return False
        
        try:
            # Make sure event_data has a timestamp
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now()
            elif isinstance(event_data['timestamp'], str):
                try:
                    event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    event_data['timestamp'] = datetime.now()
                    
            # Store event in MongoDB
            result = self.events_collection.insert_one(event_data)
            
            # If this is a rating event, also update the ratings collection
            if event_data.get('event_type') == 'rate' and 'rating' in event_data.get('data', {}):
                self._update_rating(
                    user_id=event_data['user_id'],
                    location_id=event_data['location_id'],
                    rating=event_data['data']['rating'],
                    timestamp=event_data['timestamp']
                )
            
            return True
        
        except Exception as e:
            logger.error(f"Error storing event in MongoDB: {str(e)}")
            return False
    
    def _update_rating(self, user_id, location_id, rating, timestamp):
        """
        Update a user's rating for a location
        
        Parameters:
        -----------
        user_id : str or int
            User ID
        location_id : str or int
            Location ID
        rating : float
            Rating value
        timestamp : datetime
            Time of the rating
        """
        try:
            # Convert to strings to ensure consistent typing
            user_id = str(user_id)
            location_id = str(location_id)
            
            # Upsert rating
            self.ratings_collection.update_one(
                {'user_id': user_id, 'location_id': location_id},
                {'$set': {
                    'rating': float(rating),
                    'timestamp': timestamp,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error updating rating in MongoDB: {str(e)}")

    def store_recommendation(self, user_id, recommendations, context=None):
        """
        Store recommendations for a user
        
        Parameters:
        -----------
        user_id : str or int
            User ID
        recommendations : list
            List of recommendation objects
        context : dict, optional
            Context in which recommendations were generated
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # Check if we have a valid db connection
        if self.db is None or self.recommendations_collection is None:
            if not self._connect():
                return False
        
        try:
            # Store recommendations
            doc = {
                'user_id': str(user_id),
                'timestamp': datetime.now(),
                'recommendations': recommendations,
                'context': context or {}
            }
            
            self.recommendations_collection.insert_one(doc)
            return True
        
        except Exception as e:
            logger.error(f"Error storing recommendations in MongoDB: {str(e)}")
            return False

    def get_user_events(self, user_id, limit=50):
        """
        Get recent events for a user
        
        Parameters:
        -----------
        user_id : str or int
            User ID
        limit : int
            Maximum number of events to retrieve
            
        Returns:
        --------
        list
            List of events
        """
        # Check if we have a valid db connection
        if self.db is None or self.events_collection is None:
            if not self._connect():
                return []
        
        try:
            # Convert to string to ensure consistent typing
            user_id = str(user_id)
            
            # Get recent events for user
            cursor = self.events_collection.find(
                {'user_id': user_id}
            ).sort('timestamp', -1).limit(limit)
            
            return list(cursor)
        
        except Exception as e:
            logger.error(f"Error retrieving user events from MongoDB: {str(e)}")
            return []
    
    def get_user_ratings(self, user_id):
        """
        Get all ratings for a user
        
        Parameters:
        -----------
        user_id : str or int
            User ID
            
        Returns:
        --------
        list
            List of ratings
        """
        if self.db is None or self.ratings_collection is None:
            if not self._connect():
                return []
        
        try:
            # Convert to string to ensure consistent typing
            user_id = str(user_id)
            
            # Get ratings for user
            cursor = self.ratings_collection.find({'user_id': user_id})
            
            return list(cursor)
        
        except Exception as e:
            logger.error(f"Error retrieving user ratings from MongoDB: {str(e)}")
            return []
    
    def export_events_to_csv(self, output_path=None):
        """
        Export events from MongoDB to CSV
        
        Parameters:
        -----------
        output_path : str or None
            Path to save the CSV file. If None, uses default path.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.db is None or self.events_collection is None:
            if not self._connect():
                return False
                
        try:
            if output_path is None:
                output_path = os.path.join(Config.RAW_DATA_DIR, 'location_events.csv')
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get all events
            events = list(self.events_collection.find({}))
            
            if not events:
                logger.warning("No events found in MongoDB")
                return False
                
            # Convert ObjectId to string
            for event in events:
                if '_id' in event:
                    event['_id'] = str(event['_id'])
                    
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Make sure required columns exist
            required_columns = ['user_id', 'location_id', 'event_type', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'location_id' and 'product_id' in df.columns:
                        df['location_id'] = df['product_id']
                    elif col == 'timestamp':
                        df[col] = datetime.now()
                    else:
                        df[col] = ''
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} events to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting events to CSV: {str(e)}")
            return False
            
    def export_ratings_to_csv(self, output_path=None):
        """
        Export ratings from MongoDB to CSV
        
        Parameters:
        -----------
        output_path : str or None
            Path to save the CSV file. If None, uses default path.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.db is None or self.ratings_collection is None:
            if not self._connect():
                return False
                
        try:
            if output_path is None:
                output_path = os.path.join(Config.RAW_DATA_DIR, 'location_ratings.csv')
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get all ratings
            ratings = list(self.ratings_collection.find({}))
            
            if not ratings:
                logger.warning("No ratings found in MongoDB")
                return False
                
            # Convert ObjectId to string
            for rating in ratings:
                if '_id' in rating:
                    rating['_id'] = str(rating['_id'])
                    
            # Convert to DataFrame
            df = pd.DataFrame(ratings)
            
            # Make sure required columns exist
            required_columns = ['user_id', 'location_id', 'rating', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'location_id' and 'product_id' in df.columns:
                        df['location_id'] = df['product_id']
                    elif col == 'timestamp':
                        df[col] = datetime.now()
                    else:
                        df[col] = ''
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} ratings to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting ratings to CSV: {str(e)}")
            return False
    
    def close(self):
        """
        Close the MongoDB connection
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
