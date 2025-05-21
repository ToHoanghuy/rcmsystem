# Import CSV data to MongoDB
# This script imports data from CSV files into MongoDB

import os
import sys
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import project modules
from database.mongodb import MongoDB
from config.config import Config
from utils.mongodb_store import MongoDBEventStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_locations_to_mongodb():
    """
    Import locations from CSV to MongoDB
    """
    print("\n=== Importing Locations to MongoDB ===")
    
    try:
        # Initialize MongoDB connection
        mongo_db = MongoDB()
        
        # Get CSV file path
        csv_path = os.path.join(Config.RAW_DATA_DIR, 'locations.csv')
        fallback_path = os.path.join(Config.RAW_DATA_DIR, 'products.csv')
        
        if not os.path.exists(csv_path) and os.path.exists(fallback_path):
            print(f"Using fallback file: {fallback_path}")
            csv_path = fallback_path
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        # Read CSV
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines='skip')
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
        
        # Fix any issues with the data
        for col in df.columns:
            if df[col].dtype == 'object':
                # Replace NaN with empty string
                df[col] = df[col].fillna('')
        
        # Ensure we have a product_id column
        if 'product_id' not in df.columns:
            if 'id' in df.columns:
                df['product_id'] = df['id']
            elif '_id' in df.columns:
                df['product_id'] = df['_id']
            else:
                # Generate product_id as index
                df['product_id'] = df.index.astype(str)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Import records
        result = mongo_db.places.insert_many(records)
        
        print(f"✅ Successfully imported {len(result.inserted_ids)} records to MongoDB")
        return True
    
    except Exception as e:
        print(f"❌ Error importing locations to MongoDB: {str(e)}")
        return False

def import_events_to_mongodb():
    """
    Import events from CSV to MongoDB
    """
    print("\n=== Importing Events to MongoDB ===")
    
    try:
        # Initialize MongoDB event store
        event_store = MongoDBEventStore()
        
        # Get CSV file path
        csv_path = os.path.join(Config.RAW_DATA_DIR, 'location_events.csv')
        fallback_path = os.path.join(Config.RAW_DATA_DIR, 'events.csv')
        
        if not os.path.exists(csv_path) and os.path.exists(fallback_path):
            print(f"Using fallback file: {fallback_path}")
            csv_path = fallback_path
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        # Read CSV
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"Read {len(df)} records from {csv_path}")
        
        # Get current count
        current_count = event_store.events_collection.count_documents({})
        print(f"Current count in MongoDB: {current_count}")
        
        # Ask user if they want to import
        if current_count > 0:
            response = input(f"There are already {current_count} records in MongoDB. Import anyway? (y/n): ")
            if response.lower() != 'y':
                print("Import cancelled.")
                return False
        
        # Prepare events data
        events = []
        for _, row in df.iterrows():
            try:
                # Construct event object
                event = {
                    'user_id': str(row.get('user_id', '')),
                    'location_id': str(row.get('product_id', row.get('location_id', ''))), 
                    'event_type': row.get('event_type', 'view'),
                    'timestamp': datetime.now().isoformat(),
                    'data': {}
                }
                
                # Add additional data if available
                for col in row.index:
                    if col not in ['user_id', 'product_id', 'location_id', 'event_type', 'timestamp']:
                        event['data'][col] = row[col]
                
                events.append(event)
            except Exception as e:
                print(f"Error processing event: {str(e)}")
        
        # Import records
        print(f"Importing {len(events)} events to MongoDB...")
        
        success_count = 0
        for event in events:
            try:
                result = event_store.store_event(event)
                if result:
                    success_count += 1
            except Exception as e:
                print(f"Error storing event: {str(e)}")
        
        print(f"✅ Successfully imported {success_count} events to MongoDB")
        return True
    
    except Exception as e:
        print(f"❌ Error importing events to MongoDB: {str(e)}")
        return False

def import_ratings_to_mongodb():
    """
    Import ratings from CSV to MongoDB
    """
    print("\n=== Importing Ratings to MongoDB ===")
    
    try:
        # Initialize MongoDB event store
        event_store = MongoDBEventStore()
        
        # Get CSV file path
        csv_path = os.path.join(Config.RAW_DATA_DIR, 'location_ratings.csv')
        fallback_path = os.path.join(Config.RAW_DATA_DIR, 'dataset.csv')
        
        if not os.path.exists(csv_path) and os.path.exists(fallback_path):
            print(f"Using fallback file: {fallback_path}")
            csv_path = fallback_path
        
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return False
        
        # Read CSV
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"Read {len(df)} records from {csv_path}")
        
        # Get current count
        current_count = event_store.ratings_collection.count_documents({})
        print(f"Current count in MongoDB: {current_count}")
        
        # Ask user if they want to import
        if current_count > 0:
            response = input(f"There are already {current_count} records in MongoDB. Import anyway? (y/n): ")
            if response.lower() != 'y':
                print("Import cancelled.")
                return False
        
        # Map column names to expected format
        column_mapping = {
            'user': 'user_id',
            'item': 'location_id',
            'rating': 'rating',
            'product_id': 'location_id'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Prepare ratings data
        ratings = []
        for _, row in df.iterrows():
            if 'user_id' in row and 'location_id' in row and 'rating' in row:
                try:
                    # Construct rating object
                    rating = {
                        'user_id': str(row['user_id']),
                        'location_id': str(row['location_id']),
                        'rating': float(row['rating']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    ratings.append(rating)
                except Exception as e:
                    print(f"Error processing rating: {str(e)}")
            else:
                required = []
                if 'user_id' not in row:
                    required.append('user_id')
                if 'location_id' not in row:
                    required.append('location_id')
                if 'rating' not in row:
                    required.append('rating')
                print(f"Skipping record, missing required fields: {', '.join(required)}")
        
        # Import records
        print(f"Importing {len(ratings)} ratings to MongoDB...")
        
        success_count = 0
        for rating in ratings:
            try:
                # Use upsert to avoid duplicates
                result = event_store.ratings_collection.update_one(
                    {'user_id': rating['user_id'], 'location_id': rating['location_id']},
                    {'$set': rating},
                    upsert=True
                )
                if result.acknowledged:
                    success_count += 1
            except Exception as e:
                print(f"Error storing rating: {str(e)}")
        
        print(f"✅ Successfully imported {success_count} ratings to MongoDB")
        return True
    
    except Exception as e:
        print(f"❌ Error importing ratings to MongoDB: {str(e)}")
        return False

def show_menu():
    """
    Show the main menu
    """
    print("\n" + "="*50)
    print(" CSV TO MONGODB IMPORT MENU")
    print("="*50)
    print("1. Import Locations")
    print("2. Import Events")
    print("3. Import Ratings")
    print("4. Import All")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    return choice

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" CSV TO MONGODB IMPORT TOOL")
    print("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Import all data
        import_locations_to_mongodb()
        import_events_to_mongodb()
        import_ratings_to_mongodb()
    else:
        # Show interactive menu
        while True:
            choice = show_menu()
            
            if choice == '1':
                import_locations_to_mongodb()
            elif choice == '2':
                import_events_to_mongodb()
            elif choice == '3':
                import_ratings_to_mongodb()
            elif choice == '4':
                import_locations_to_mongodb()
                import_events_to_mongodb()
                import_ratings_to_mongodb()
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")
