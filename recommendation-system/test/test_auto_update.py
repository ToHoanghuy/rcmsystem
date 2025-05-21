"""
Test script for MongoDB change stream and recommendation system auto-update

This script demonstrates how to:
1. Add a new location to MongoDB
2. Watch for automatic model updates in the recommendation system
"""

import requests
import json
import time
import pandas as pd
import os
import sys
from datetime import datetime

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb import MongoDB
from config.config import Config

# Define the base URL for the recommendation API
BASE_URL = "http://localhost:5000"

def test_mongodb_monitor():
    """Test MongoDB monitor and auto-update functionality"""
    print("\n===== Testing MongoDB Monitor and Auto-Update =====")
    
    # Connect to MongoDB
    mongo = MongoDB()
    
    # 1. Check current number of locations
    places_count_before = mongo.places.count_documents({})
    print(f"Current number of places in MongoDB: {places_count_before}")
    
    # 2. Get current recommendations for a test user
    try:
        response = requests.get(f"{BASE_URL}/recommend?user_id=123&case=content_based&product_id=1")
        response.raise_for_status()
        print("Current recommendation sample:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error getting recommendations: {e}")
    
    # 3. Create a new location to add to MongoDB
    new_location = {
        "product_id": f"test_{int(time.time())}",  # Generate unique ID
        "product_name": "Test Auto-Update Location",
        "description": "This location was added to test the auto-update feature",
        "province": "Auto Update Province",
        "services": ["testing", "auto-update", "monitoring"],
        "capacity": 100,
        "price_range": "medium",
        "created_at": datetime.now().isoformat()
    }
    
    # 4. Add the location to MongoDB
    print("\nAdding new location to MongoDB...")
    result = mongo.places.insert_one(new_location)
    print(f"Added location with ID: {result.inserted_id}")
    
    # 5. Wait for the monitor to detect changes and update the model
    print("\nWaiting for automatic model update (this may take a few seconds)...")
    time.sleep(10)  # Wait a bit for the update to happen
    
    # 6. Verify the update was applied by checking monitor status
    try:
        response = requests.post(f"{BASE_URL}/admin/monitor?action=status")
        response.raise_for_status()
        print("Monitor status:", response.json())
    except Exception as e:
        print(f"Error checking monitor status: {e}")
    
    # 7. Force an update if needed
    try:
        response = requests.post(f"{BASE_URL}/admin/monitor?action=force_update")
        response.raise_for_status()
        print("Forced update status:", response.json())
    except Exception as e:
        print(f"Error forcing update: {e}")
    
    # 8. Get recommendations again to see if they include the new location
    time.sleep(5)  # Wait a bit for the update to finish
    try:
        response = requests.get(f"{BASE_URL}/recommend?user_id=123&case=content_based&product_id={new_location['product_id']}")
        response.raise_for_status()
        print("\nUpdated recommendation for new location:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error getting recommendations after update: {e}")
    
    # 9. Clean up - remove the test location
    print("\nCleaning up - removing test location...")
    mongo.places.delete_one({"product_id": new_location["product_id"]})
    
    print("\n===== MongoDB Monitor and Auto-Update Test Completed =====")

def test_import_data():
    """Test the data import API and auto-update functionality"""
    print("\n===== Testing Data Import API =====")
    
    # Create a temporary CSV file with test locations
    test_data = pd.DataFrame([
        {
            "product_id": f"import_test_1_{int(time.time())}",
            "product_name": "Import Test Location 1",
            "description": "This is a test location for the import API",
            "province": "Import Test Province",
            "services": ["api_testing", "import_test"],
            "capacity": 200
        },
        {
            "product_id": f"import_test_2_{int(time.time())}",
            "product_name": "Import Test Location 2",
            "description": "Another test location for the import API",
            "province": "Import Test Province",
            "services": ["api_testing", "import_test"], 
            "capacity": 150
        }
    ])
    
    # Save to a temporary CSV file
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
    test_data_path = temp_file.name
    temp_file.close()
    
    test_data.to_csv(test_data_path, index=False)
    print(f"Created test data CSV at: {test_data_path}")
    
    # Upload the CSV file using the import API
    try:
        with open(test_data_path, 'rb') as f:
            files = {'file': (os.path.basename(test_data_path), f, 'text/csv')}
            data = {'collection': 'places'}
            response = requests.post(f"{BASE_URL}/admin/import", files=files, data=data)
            response.raise_for_status()
            print("Import API response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error importing data: {e}")
    
    # Wait for the updates to be applied
    print("\nWaiting for model to update after import...")
    time.sleep(10)
    
    # Check recommendations for one of the imported locations
    try:
        response = requests.get(f"{BASE_URL}/recommend?user_id=123&case=content_based&product_id={test_data['product_id'][0]}")
        response.raise_for_status()
        print("\nRecommendation for imported location:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error getting recommendations for imported location: {e}")
    
    # Clean up
    os.unlink(test_data_path)
    print(f"Cleaned up test file: {test_data_path}")
    
    print("\n===== Data Import Test Completed =====")

if __name__ == "__main__":
    # Run the tests
    print("Starting MongoDB Monitor and Data Import tests...")
    
    try:
        test_mongodb_monitor()
        test_import_data()
    except Exception as e:
        print(f"Error during tests: {e}")
        
    print("\nTests completed!")
