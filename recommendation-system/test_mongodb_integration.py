# MongoDB Integration Test Script
# This script tests the MongoDB integration for the recommendation system

import pandas as pd
import logging
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import project modules
from database.mongodb import MongoDB
from config.config import Config
from utils.mongodb_store import MongoDBEventStore
from utils.mongodb_loader import load_data_from_mongodb_or_csv

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
        
        # Check if db is initialized properly
        if event_store.db is None:
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
            if collection is not None:
                count = collection.count_documents({})
                print(f"  - {name}: {count} documents")
            else:
                print(f"  - {name}: Not initialized")
        
        # Test storing an event
        test_event = {
            'user_id': 'test_user',
            'location_id': 'test_location',
            'event_type': 'test',
            'timestamp': datetime.now(),
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

def test_mongodb_data_loading():
    """
    Test loading data from MongoDB
    """
    print("\n=== Testing MongoDB Data Loading ===")
    
    # Load locations
    print("\nLoading locations from MongoDB...")
    locations_df = load_data_from_mongodb_or_csv('locations')
    print(f"Loaded {len(locations_df)} locations")
    if not locations_df.empty:
        print("✅ Successfully loaded locations from MongoDB")
        print("Sample location data:")
        print(locations_df.head(2))
        print(f"Columns: {locations_df.columns.tolist()}")
    else:
        print("❌ No locations data found in MongoDB")
    
    # Load events
    print("\nLoading events from MongoDB...")
    events_df = load_data_from_mongodb_or_csv('events')
    print(f"Loaded {len(events_df)} events")
    if not events_df.empty:
        print("✅ Successfully loaded events from MongoDB")
        print("Sample event data:")
        print(events_df.head(2))
        print(f"Columns: {events_df.columns.tolist()}")
    else:
        print("❌ No events data found in MongoDB")
    
    # Load ratings
    print("\nLoading ratings from MongoDB...")
    ratings_df = load_data_from_mongodb_or_csv('ratings')
    print(f"Loaded {len(ratings_df)} ratings")
    if not ratings_df.empty:
        print("✅ Successfully loaded ratings from MongoDB")
        print("Sample rating data:")
        print(ratings_df.head(2))
        print(f"Columns: {ratings_df.columns.tolist()}")
    else:
        print("❌ No ratings data found in MongoDB")
    
    # Check if we have enough data for recommendations
    has_enough_data = (not locations_df.empty) and (not ratings_df.empty or not events_df.empty)
    
    if has_enough_data:
        print("\n✅ MongoDB has sufficient data for recommendations")
    else:
        print("\n❌ MongoDB does not have sufficient data for recommendations")
        print("   The system will use fallback CSV data")
    
    return has_enough_data

def test_import_csv_to_mongodb():
    """
    Test importing CSV data to MongoDB
    """
    print("\n=== Testing CSV Import to MongoDB ===")
    
    try:
        # Initialize MongoDB connection
        mongo_db = MongoDB()
        
        # Import locations
        print("\nImporting locations.csv to MongoDB...")
        csv_path = os.path.join(Config.RAW_DATA_DIR, 'locations.csv')
        
        if os.path.exists(csv_path):
            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"Read {len(df)} records from {csv_path}")
            
            # Get current count
            current_count = mongo_db.places.count_documents({})
            print(f"Current count in MongoDB: {current_count}")
            
            # Ask user if they want to import
            if current_count > 0:
                response = input(f"There are already {current_count} records in MongoDB. Import anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Import cancelled.")
                    return False
            
            # Import records
            records = df.to_dict('records')
            result = mongo_db.places.insert_many(records)
            
            print(f"✅ Successfully imported {len(result.inserted_ids)} records to MongoDB")
            return True
        else:
            print(f"❌ CSV file not found: {csv_path}")
            return False
    
    except Exception as e:
        print(f"❌ Error importing CSV to MongoDB: {str(e)}")
        return False

def show_menu():
    """
    Show the main menu
    """
    print("\n" + "="*50)
    print(" MONGODB INTEGRATION TEST MENU")
    print("="*50)
    print("1. Test MongoDB Connection")
    print("2. Test MongoDB Event Store")
    print("3. Test MongoDB Data Loading")
    print("4. Import CSV Data to MongoDB")
    print("5. Run All Tests")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    return choice

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" MONGODB INTEGRATION TEST")
    print("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Run all tests
        connection_ok = test_mongodb_connection()
        
        if connection_ok:
            store_ok = test_mongodb_store()
            data_ok = test_mongodb_data_loading()
            
            # Summary
            print("\n" + "="*50)
            print(" TEST SUMMARY")
            print("="*50)
            print(f"MongoDB Connection: {'✅ Success' if connection_ok else '❌ Failed'}")
            print(f"MongoDB Event Store: {'✅ Success' if store_ok else '❌ Failed'}")
            print(f"MongoDB Data Loading: {'✅ Success' if data_ok else '❌ Warning - Using fallback data'}")
            
            if connection_ok and store_ok:
                print("\n✅ The system is ready to use MongoDB integration")
                print("   Run 'python main_mongodb.py' to start the system with MongoDB data")
            else:
                print("\n⚠️ There are issues with the MongoDB integration")
                print("   The system will use fallback CSV data")
        else:
            print("\n❌ Failed to connect to MongoDB")
            print("   Please check your MongoDB connection string in .env file")
            print("   The system will use fallback CSV data")
    else:
        # Show interactive menu
        while True:
            choice = show_menu()
            
            if choice == '1':
                test_mongodb_connection()
            elif choice == '2':
                test_mongodb_store()
            elif choice == '3':
                test_mongodb_data_loading()
            elif choice == '4':
                test_import_csv_to_mongodb()
            elif choice == '5':
                connection_ok = test_mongodb_connection()
                
                if connection_ok:
                    store_ok = test_mongodb_store()
                    data_ok = test_mongodb_data_loading()
                    
                    # Summary
                    print("\n" + "="*50)
                    print(" TEST SUMMARY")
                    print("="*50)
                    print(f"MongoDB Connection: {'✅ Success' if connection_ok else '❌ Failed'}")
                    print(f"MongoDB Event Store: {'✅ Success' if store_ok else '❌ Failed'}")
                    print(f"MongoDB Data Loading: {'✅ Success' if data_ok else '❌ Warning - Using fallback data'}")
                    
                    if connection_ok and store_ok:
                        print("\n✅ The system is ready to use MongoDB integration")
                        print("   Run 'python main_mongodb.py' to start the system with MongoDB data")
                    else:
                        print("\n⚠️ There are issues with the MongoDB integration")
                        print("   The system will use fallback CSV data")
                else:
                    print("\n❌ Failed to connect to MongoDB")
                    print("   Please check your MongoDB connection string in .env file")
                    print("   The system will use fallback CSV data")
            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")