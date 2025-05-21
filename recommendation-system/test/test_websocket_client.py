"""
WebSocket client test for real-time recommendations.
This script simulates multiple clients connecting to the recommendation system
and generates events to test the real-time recommendation functionality.
"""

import sys
import os
import time
import json
import random
import asyncio
import socketio
import requests
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration
API_URL = 'http://localhost:5000'
NUM_USERS = 3
NUM_EVENTS_PER_USER = 5
EVENT_TYPES = ['view', 'click', 'book', 'rate', 'favorite']
EVENT_DELAY = 1  # seconds between events

# Create async Socket.IO clients
sio_clients = {}

# Store recommendations received
recommendations = {}

async def connect_user(user_id):
    """Connect a user to the WebSocket server and register them"""
    sio = socketio.AsyncClient()
    
    @sio.event
    async def connect():
        print(f"User {user_id} connected to Socket.IO")
        await sio.emit('register_user', {'user_id': user_id})
    
    @sio.event
    async def disconnect():
        print(f"User {user_id} disconnected")
    
    @sio.event
    async def registration_success(data):
        print(f"User {user_id} registered successfully: {data['message']}")
    
    @sio.event
    async def registration_failed(data):
        print(f"User {user_id} registration failed: {data['message']}")
    
    @sio.event
    async def realtime_recommendation(data):
        print(f"User {user_id} received recommendations: {len(data['recommendations'])} items")
        recommendations[user_id] = data['recommendations']
        # Print first recommendation
        if data['recommendations']:
            first_rec = data['recommendations'][0]
            print(f"  First recommendation: {first_rec.get('name', 'Unknown')} - Score: {first_rec.get('score', 'N/A')}")
    
    try:
        await sio.connect(API_URL)
        sio_clients[user_id] = sio
        return True
    except Exception as e:
        print(f"Error connecting user {user_id}: {str(e)}")
        return False

async def disconnect_user(user_id):
    """Disconnect a user from the WebSocket server"""
    if user_id in sio_clients:
        try:
            await sio_clients[user_id].disconnect()
            del sio_clients[user_id]
            print(f"User {user_id} disconnected")
        except Exception as e:
            print(f"Error disconnecting user {user_id}: {str(e)}")

def send_event(user_id, location_id, event_type):
    """Send an event to the recommendation system via HTTP API"""
    event_data = {
        'user_id': user_id,
        'location_id': location_id,
        'event_type': event_type,
        'data': {}
    }
    
    # Add rating if event type is 'rate'
    if event_type == 'rate':
        event_data['data']['rating'] = random.randint(1, 5)
    
    try:
        response = requests.post(f"{API_URL}/api/track", json=event_data)
        if response.status_code == 200:
            result = response.json()
            print(f"Event sent: User {user_id} {event_type} Location {location_id} - Success: {result.get('success', False)}")
            return True
        else:
            print(f"Failed to send event: {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"Error sending event: {str(e)}")
        return False

async def run_user_simulation(user_id):
    """Run a full simulation for a single user"""
    # Connect user to WebSocket
    if not await connect_user(user_id):
        return
    
    # Wait a bit for connection to stabilize
    await asyncio.sleep(1)
    
    # Generate events for this user
    location_ids = [100 + i for i in range(10)]  # Example location IDs from 100-109
    
    for i in range(NUM_EVENTS_PER_USER):
        # Randomly select event type and location
        event_type = random.choice(EVENT_TYPES)
        location_id = random.choice(location_ids)
        
        # Send event
        send_event(user_id, location_id, event_type)
        
        # Wait between events
        await asyncio.sleep(EVENT_DELAY)
    
    # Keep connection open for a bit to receive recommendations
    await asyncio.sleep(2)
    
    # Disconnect user
    await disconnect_user(user_id)

async def run_test():
    """Run the full WebSocket test with multiple users"""
    print(f"Starting WebSocket test with {NUM_USERS} users, {NUM_EVENTS_PER_USER} events each")
    
    # Create tasks for each user
    tasks = [run_user_simulation(f"test_user_{i}") for i in range(1, NUM_USERS + 1)]
    
    # Run all user simulations in parallel
    await asyncio.gather(*tasks)
    
    # Print summary
    print("\nTest completed!")
    print(f"Users tested: {NUM_USERS}")
    print(f"Total events sent: {NUM_USERS * NUM_EVENTS_PER_USER}")
    print(f"Recommendations received: {len(recommendations)} users")
    
    # Write results to file
    with open('websocket_test_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'users_tested': NUM_USERS,
            'events_per_user': NUM_EVENTS_PER_USER,
            'recommendations_received': {k: len(v) for k, v in recommendations.items()}
        }, f, indent=2)

if __name__ == "__main__":
    print("WebSocket Real-Time Recommendation Test")
    print("======================================")
    
    # Check server availability
    try:
        response = requests.get(f"{API_URL}/api/recommend?user_id=1&case=hybrid")
        if response.status_code == 200:
            print(f"Server is available at {API_URL}")
        else:
            print(f"Server returned unexpected status code: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Server is not available at {API_URL}: {str(e)}")
        print("Please make sure the recommendation server is running")
        sys.exit(1)
    
    # Run async test
    asyncio.run(run_test())
