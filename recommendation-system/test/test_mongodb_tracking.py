"""
Test script to verify the fixes for MongoDB-style IDs in the recommendation system.
This script simulates a request with MongoDB IDs and checks if the system handles it correctly.
"""
import os
import sys
import json
import requests
from datetime import datetime

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def test_mongodb_id_tracking():
    """Test tracking an event with MongoDB-style IDs"""
    print("\n=== Testing Event Tracking with MongoDB IDs ===\n")
    
    # MongoDB-style IDs
    user_id = "6729d0e8dda9481629a2a2e9"
    location_id = "6704f3650722c4f99305dc5d"
    
    # Create event data
    event_data = {
        "user_id": user_id,
        "location_id": location_id,
        "event_type": "view",
        "timestamp": datetime.now().isoformat(),
        "data": {"source": "test_script"}
    }
    
    print(f"Sending event: {json.dumps(event_data, indent=2)}")
    
    # Send request to the API
    try:
        response = requests.post(
            "http://localhost:5000/api/track", 
            json=event_data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ Success! The API handled MongoDB IDs correctly.")
        else:
            print("\n❌ Failed. The API returned an error.")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def test_mongodb_id_recommendations():
    """Test getting recommendations with MongoDB-style IDs"""
    print("\n=== Testing Recommendations with MongoDB IDs ===\n")
    
    # MongoDB-style ID
    user_id = "6729d0e8dda9481629a2a2e9"
    
    print(f"Getting recommendations for user: {user_id}")
    
    # Send request to the API
    try:
        response = requests.get(
            f"http://localhost:5000/api/recommend?user_id={user_id}&case=hybrid"
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            recs = result.get("recommendations", [])
            print(f"Received {len(recs)} recommendations")
            
            if len(recs) > 0:
                print("\nFirst 3 recommendations:")
                for i, rec in enumerate(recs[:3]):
                    print(f"  {i+1}. Location ID: {rec.get('location_id')} - Score: {rec.get('score')}")
                
            print("\n✅ Success! The API returned recommendations for MongoDB IDs.")
        else:
            print(f"\n❌ Failed. Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "recommendations":
        test_mongodb_id_recommendations()
    else:
        test_mongodb_id_tracking()
        print("\nTo test recommendations, run the script with the 'recommendations' argument:")
        print("python test_mongodb_tracking.py recommendations")
