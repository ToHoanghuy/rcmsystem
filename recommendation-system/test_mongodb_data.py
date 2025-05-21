# Test MongoDB data loading
from utils.mongodb_loader import load_data_from_mongodb_or_csv
import pandas as pd
import logging
from database.mongodb import MongoDB
from config.config import Config
from utils.mongodb_store import MongoDBEventStore
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mongodb_connection():
    """
    Test MongoDB connection
    """
    print("\n=== Testing MongoDB Connection ===")
    
    # Get MongoDB URI from environment or config
    mongodb_uri = os.environ.get('MONGODB_URI') or Config.MONGODB_URI
    
    if not mongodb_uri:
        print("ERROR: MONGODB_URI is not configured in Config or .env")
        return False
    
    print(f"MongoDB URI: {mongodb_uri[:15]}...{mongodb_uri[-15:] if len(mongodb_uri) > 30 else mongodb_uri[-15:]}")
    
    try:
        # Test MongoDB connection
        mongo_db = MongoDB()
        connected = mongo_db.test_connection()
        
        if connected:
            db_name = mongo_db.db.name
            collections = mongo_db.db.list_collection_names()
            
            print(f"✅ Successfully connected to MongoDB database: {db_name}")
            print(f"Available collections: {', '.join(collections)}")
            
            # Test data in collections
            for collection_name in collections:
                count = mongo_db.db[collection_name].count_documents({})
                print(f"  - {collection_name}: {count} documents")
            
            return True
        else:
            print("❌ Failed to connect to MongoDB")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {str(e)}")
        return False

def test_mongodb_store():
    """
    Test MongoDB event store
    """
    print("\n=== Testing MongoDB Event Store ===")
    
    try:
        # Initialize MongoDB event store
        event_store = MongoDBEventStore()
        
        if not event_store.db:
            print("❌ Failed to connect to MongoDB via event store")
            return False
            
        db_name = event_store.db.name
        print(f"✅ Successfully connected to MongoDB database: {db_name}")
        
        # Check collections
        collections = {
            'events': event_store.events_collection,
            'ratings': event_store.ratings_collection,
            'recommendations': event_store.recommendations_collection
        }
        
        for name, collection in collections.items():
            if collection:
                count = collection.count_documents({})
                print(f"  - {name}: {count} documents")
            else:
                print(f"  - {name}: Not initialized")
        
        # Test storing an event
        test_event = {
            'user_id': 'test_user',
            'location_id': 'test_location',
            'event_type': 'test',
            'timestamp': pd.Timestamp.now(),
            'data': {
                'test': True
            }
        }
        
        print("\nTesting event storage...")
        success = event_store.store_event(test_event)
        
        if success:
            print("✅ Successfully stored test event in MongoDB")
            
            # Retrieve the event to confirm
            events = event_store.get_user_events('test_user', limit=1)
            if events and len(events) > 0:
                print("✅ Successfully retrieved test event from MongoDB")
                # Clean up the test event
                event_store.events_collection.delete_one({'user_id': 'test_user', 'event_type': 'test'})
                print("✅ Successfully cleaned up test event")
            else:
                print("❌ Failed to retrieve test event from MongoDB")
        else:
            print("❌ Failed to store test event in MongoDB")
        
        return success
    
    except Exception as e:
        print(f"❌ Error testing MongoDB event store: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_data_loading()
