"""
Test script to verify that MongoDB-style IDs work with the recommendation system
"""
import os
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mongodb_id_event():
    """
    Test event tracking with a MongoDB-style ID
    """
    print("Testing event tracking with MongoDB-style IDs...")

    # Define a MongoDB-style ID and create a test event
    test_user_id = "6729d0e8dda9481629a2a2e9"  # MongoDB-style hexadecimal ID
    test_location_id = "5f8d0e8dda9481629a2a3b7"  # Another MongoDB-style ID
    
    # Create test event data
    event_data = {
        "user_id": test_user_id,
        "location_id": test_location_id,
        "event_type": "view",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "device_info": "Test device"
        }
    }

    # Get data directory relative to this script
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Import the save_event_to_csv function
    from services.api_endpoints import save_event_to_csv
    
    # Save the event
    result = save_event_to_csv(event_data, DATA_DIR)
    print(f"Event saved: {result}")
    
    # Try to read it back from CSV
    events_file = os.path.join(DATA_DIR, 'raw', 'location_events.csv')
    if os.path.exists(events_file):
        events_df = pd.read_csv(events_file)
        print("\nEvents from CSV:")
        print(events_df.tail())
        
        # Verify the MongoDB ID was saved correctly
        user_events = events_df[events_df['user_id'] == str(test_user_id)]
        if not user_events.empty:
            print(f"\nFound {len(user_events)} events for user {test_user_id}")
        else:
            print(f"\nError: No events found for user {test_user_id}")
    else:
        print(f"Error: Events file not found at {events_file}")

    # Check the full JSON data
    full_events_file = os.path.join(DATA_DIR, 'raw', 'location_events_full.json')
    if os.path.exists(full_events_file):
        print("\nFull event data from JSON:")
        with open(full_events_file, 'r') as f:
            for i, line in enumerate(f):
                if i > 5:  # Show just a few records
                    break
                try:
                    data = json.loads(line)
                    print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    print(f"Error parsing line: {line}")
    else:
        print(f"Full events file not found at {full_events_file}")

if __name__ == "__main__":
    test_mongodb_id_event()
