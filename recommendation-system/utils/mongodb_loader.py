"""
MongoDB data loader module for the recommendation system.
This module provides functions to load data from MongoDB collections instead of CSV files.
"""

import os
import logging
import pandas as pd
from datetime import datetime
from database.mongodb import MongoDB
from config.config import Config
import json
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_mongodb_locations():
    """
    Load location data from MongoDB instead of CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing location data
    """
    try:
        # Initialize MongoDB connection
        mongo_db = MongoDB()
        
        # Get all places from MongoDB
        places = mongo_db.get_all_places()
        
        if not places:
            logger.warning("No location data found in MongoDB")
            return pd.DataFrame({'product_id': ['dummy_product_1']})
            
        # Convert to DataFrame and handle ObjectId
        from bson import ObjectId, json_util
        
        # Convert ObjectId to string in the raw data
        for place in places:
            for key, value in place.items():
                if isinstance(value, ObjectId):
                    place[key] = str(value)
                elif isinstance(value, list):
                    # Convert any ObjectId in lists
                    for i in range(len(value)):
                        if isinstance(value[i], ObjectId):
                            value[i] = str(value[i])
                elif isinstance(value, dict):
                    # Convert any ObjectId in nested dicts
                    for k, v in value.items():
                        if isinstance(v, ObjectId):
                            value[k] = str(v)
                    
        # Now convert to DataFrame
        df = pd.DataFrame(places)
        
        # Ensure MongoDB _id is converted to string and mapped to product_id
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
            
            # Add product_id column if it doesn't exist
            if 'product_id' not in df.columns:
                df['product_id'] = df['_id']
          # Convert all dict/list columns to string for content-based filtering compatibility
        for col in df.columns:
            try:
                # Sử dụng ensure_ascii=False để giữ nguyên các ký tự Unicode
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            except Exception as e:
                logger.warning(f"Failed to convert column {col} to string: {str(e)}. Setting to string representation.")
                # Chuyển đổi đúng định dạng UTF-8
                df[col] = df[col].apply(lambda x: str(x).encode('utf-8', errors='ignore').decode('utf-8') if x is not None else x)
        
        # CRITICAL: Always ensure product_id exists after all transformations
        if 'product_id' not in df.columns:
            if '_id' in df.columns:
                df['product_id'] = df['_id']
            else:
                # Try to create a unique identifier if _id doesn't exist
                df['product_id'] = [f"loc_{i}" for i in range(len(df))]
        
        logger.info(f"[DEBUG] MongoDB locations columns: {df.columns.tolist()}")
        logger.info(f"Loaded {len(df)} locations from MongoDB")
        return df
        
    except Exception as e:
        logger.error(f"Error loading location data from MongoDB: {str(e)}")
        # Always return a DataFrame with at least product_id column
        return pd.DataFrame({'product_id': ['dummy_product_1']})

def load_mongodb_events():
    """
    Load user event data from MongoDB instead of CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing user events
    """
    try:
        from utils.mongodb_store import MongoDBEventStore
        
        # Initialize MongoDB event store
        event_store = MongoDBEventStore()
        
        # Get all events from collection
        events_collection = event_store.db['events']
        all_events = list(events_collection.find())
        
        if not all_events:
            logger.warning("No event data found in MongoDB")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        
        # Convert MongoDB _id to string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
            
        # Convert location_id to product_id for consistency
        if 'location_id' in df.columns and 'product_id' not in df.columns:
            df['product_id'] = df['location_id']
            
        # Convert MongoDB ObjectId to string (if present in any column)
        from bson import ObjectId
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].apply(lambda x: str(x) if isinstance(x, ObjectId) else x)
        
        logger.info(f"Loaded {len(df)} events from MongoDB")
        return df
        
    except Exception as e:
        logger.error(f"Error loading event data from MongoDB: {str(e)}")
        return pd.DataFrame()

def load_mongodb_ratings():
    """
    Load user ratings data from MongoDB instead of CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing user ratings
    """
    try:
        from utils.mongodb_store import MongoDBEventStore
        
        # Initialize MongoDB event store
        event_store = MongoDBEventStore()
        
        # Get all ratings from collection
        ratings_collection = event_store.db['ratings']
        all_ratings = list(ratings_collection.find())
        
        if not all_ratings:
            logger.warning("No rating data found in MongoDB")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(all_ratings)
        
        # Convert MongoDB _id to string
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
            
        # Ensure consistent column naming
        column_mapping = {
            'location_id': 'product_id',
            'rating': 'rating'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Add required columns if they don't exist
        required_columns = ['user_id', 'product_id', 'rating', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = datetime.now()
                else:
                    df[col] = ''
        
        logger.info(f"Loaded {len(df)} ratings from MongoDB")
        return df
        
    except Exception as e:
        logger.error(f"Error loading rating data from MongoDB: {str(e)}")
        return pd.DataFrame()

def get_fallback_csv_path(collection_type):
    """
    Get fallback CSV path if MongoDB data is not available
    
    Parameters:
    -----------
    collection_type : str
        Type of collection ('locations', 'events', 'ratings')
        
    Returns:
    --------
    str
        Path to fallback CSV file
    """
    base_dir = Config.RAW_DATA_DIR
    
    if collection_type == 'locations':
        return os.path.join(base_dir, 'locations.csv')
    elif collection_type == 'events':
        return os.path.join(base_dir, 'location_events.csv')
    elif collection_type == 'ratings':
        return os.path.join(base_dir, 'location_ratings.csv')
    else:
        raise ValueError(f"Unknown collection type: {collection_type}")

def load_data_from_mongodb_or_csv(collection_type):
    """
    Load data from MongoDB first, fall back to CSV if MongoDB data is not available
    
    Parameters:
    -----------
    collection_type : str
        Type of collection ('locations', 'events', 'ratings')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the requested data
    """
    # First try to load from MongoDB
    if collection_type == 'locations':
        df = load_mongodb_locations()
    elif collection_type == 'events':
        df = load_mongodb_events()
    elif collection_type == 'ratings':
        df = load_mongodb_ratings()
    else:
        raise ValueError(f"Unknown collection type: {collection_type}")
        
    # If MongoDB data is empty, fall back to CSV
    if df.empty:
        logger.warning(f"No {collection_type} data found in MongoDB, falling back to CSV")
        csv_path = get_fallback_csv_path(collection_type)
        
        if os.path.exists(csv_path):
            try:
                # Try to read the CSV file with error handling
                # on_bad_lines parameter replaces the deprecated error_bad_lines
                df = pd.read_csv(csv_path, on_bad_lines='skip')
                logger.info(f"Loaded {len(df)} records from fallback CSV: {csv_path}")
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
                # Return empty DataFrame if CSV reading fails
                df = pd.DataFrame()
        else:
            logger.warning(f"Fallback CSV not found: {csv_path}")
            
    return df

def load_ratings_from_mongodb():
    """
    Load ratings from MongoDB Review collection using the MongoDB class and config.
    Returns: DataFrame with columns: user_id, location_id, product_id, rating, timestamp
    """
    import pandas as pd
    from datetime import datetime
    from config.config import Config
    from database.mongodb import MongoDB
    try:
        mongo_db = MongoDB()
        collection_name = Config.RATINGS_COLLECTION if hasattr(Config, 'RATINGS_COLLECTION') else 'Review'
        reviews = list(mongo_db.db[collection_name].find())
        logger.info(f"[DEBUG] Fetched {len(reviews)} raw records from MongoDB collection '{collection_name}' in database '{mongo_db.db.name}'")
        if reviews:
            logger.info(f"[DEBUG] First 3 raw records: {reviews[:3]}")
        rows = []
        for r in reviews:
            sender = r.get('senderId') or r.get('user_id')
            loc = r.get('locationId') or r.get('location_id')
            rating = r.get('rating')
            date = r.get('date') or r.get('timestamp')
            # Xử lý trường hợp date là kiểu dict MongoDB
            if isinstance(date, dict) and '$date' in date:
                if isinstance(date['$date'], dict) and '$numberLong' in date['$date']:
                    try:
                        ms = int(date['$date']['$numberLong'])
                        date = datetime.utcfromtimestamp(ms / 1000)
                    except Exception:
                        date = None
                else:
                    try:
                        date = str(date['$date'])
                    except Exception:
                        date = None
            if isinstance(date, datetime):
                date = date.isoformat()
            if sender and loc and rating is not None and date:
                rows.append({
                    'user_id': str(sender),
                    'location_id': str(loc),
                    'product_id': str(loc),
                    'rating': float(rating),
                    'timestamp': date
                })
        df = pd.DataFrame(rows)
        logger.info(f"[DEBUG] DataFrame after mapping has {len(df)} rows. Columns: {list(df.columns)}")
        # Ensure both 'product_id' and 'location_id' columns exist and are consistent
        if not df.empty:
            logger.info(f"[DEBUG] First 3 mapped rows: {df.head(3).to_dict()}")
        return df
    except Exception as e:
        logger.error(f"Error loading ratings from MongoDB: {str(e)}")
        return pd.DataFrame()

def load_events_from_json(json_path=None):
    """
    Load events from location_events_full.json
    Returns: DataFrame with columns: user_id, location_id, product_id, event_type, timestamp, data
    """
    import pandas as pd
    from config.config import Config
    import json
    if json_path is None:
        json_path = os.path.join(Config.RAW_DATA_DIR, 'location_events_full.json')
    if not os.path.exists(json_path):
        logger.warning(f"Events JSON file not found: {json_path}")
        return pd.DataFrame({'user_id': [], 'location_id': [], 'product_id': [], 'event_type': [], 'timestamp': [], 'data': []})
    events = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
                # Extract location ID from either location_id or product_id field
                loc_id = event.get('location_id', event.get('product_id', ''))
                event['user_id'] = str(event.get('user_id', ''))
                event['location_id'] = str(loc_id)
                event['product_id'] = str(loc_id)  # Always ensure product_id exists and matches location_id
                event['event_type'] = event.get('event_type', '')
                event['timestamp'] = event.get('timestamp', '')
                # Convert 'data' field to string (json.dumps) to avoid unhashable dict error
                event['data'] = json.dumps(event.get('data', {}), ensure_ascii=False)
                events.append(event)
            except Exception as e:
                logger.warning(f"Error parsing event: {str(e)}")
                continue
    if not events:
        logger.warning("No events found in JSON file")
        return pd.DataFrame({'user_id': [], 'location_id': [], 'product_id': [], 'event_type': [], 'timestamp': [], 'data': []})
    df = pd.DataFrame(events)
    logger.info(f"Loaded {len(df)} real events from JSON")
    # Ensure we have all required columns
    required_cols = ['user_id', 'location_id', 'product_id', 'event_type', 'timestamp', 'data']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''
    return df[required_cols]
