import unittest
import json
from datetime import datetime

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommenders.realtime import RealtimeRecommender
from utils.realtime_integration import RealtimeIntegration


class MockSocketIO:
    """Mock SocketIO for testing"""
    
    def __init__(self):
        self.emitted_messages = []
        
    def emit(self, event, data, room=None):
        self.emitted_messages.append({
            'event': event,
            'data': data,
            'room': room
        })
        return True


class TestRealtimeRecommendations(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        # Create a realtime recommender
        self.recommender = RealtimeRecommender()
        # Create mock socket.io instance
        self.mock_socketio = MockSocketIO()
        # Create a realtime integration
        self.integration = RealtimeIntegration(self.mock_socketio, self.recommender)
        
        # Add test users and products
        self.user_id = 1
        self.product_ids = [101, 102, 103, 104, 105]
        
        # Populate with some test events
        self.recommender.record_event(self.user_id, 101, 'view')
        self.recommender.record_event(self.user_id, 102, 'view')
        self.recommender.record_event(self.user_id, 101, 'add_to_cart')
        
        # Register a user for socket communications
        self.integration.register_user(self.user_id, 'socket-id-123')
        
    def test_record_event(self):
        """Test recording events"""
        # Record a view event
        self.recommender.record_event(self.user_id, 103, 'view')
        
        # Check that the event was recorded
        self.assertIn(self.user_id, self.recommender.recent_events)
        self.assertGreaterEqual(len(self.recommender.recent_events[self.user_id]), 1)
        
        # Check that at least one event has the correct product ID
        found = False
        for event in self.recommender.recent_events[self.user_id]:
            if event['product_id'] == 103 and event['event_type'] == 'view':
                found = True
                break
        
        self.assertTrue(found)
        
    def test_cooccurrence_matrices(self):
        """Test that co-occurrence matrices are updated"""
        # Record sequential views to create co-occurrence
        self.recommender.record_event(self.user_id, 104, 'view')
        self.recommender.record_event(self.user_id, 105, 'view')
        
        # Check for co-occurrence key
        key = self.recommender._get_product_pair_key(104, 105)
        self.assertIn(key, self.recommender.coview_matrix)
        
    def test_get_realtime_recommendations(self):
        """Test getting realtime recommendations"""
        # Create some test product details
        import pandas as pd
        product_details = pd.DataFrame({
            'product_id': self.product_ids,
            'name': [f'Product {pid}' for pid in self.product_ids],
            'description': [f'Description for product {pid}' for pid in self.product_ids],
            'price': [10.0, 20.0, 30.0, 40.0, 50.0],
            'rating': [4.5, 4.0, 4.8, 3.9, 4.2]
        })
        
        self.integration.product_details = product_details
        
        # Get recommendations
        recommendations = self.recommender.get_realtime_recommendations(
            user_id=self.user_id, 
            current_product_id=101,
            event_type='view',
            product_details=product_details
        )
        
        # Check we get some recommendations
        self.assertIsInstance(recommendations, list)
        
        if len(recommendations) > 0:
            # Check recommendations have expected fields
            self.assertIn('product_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            self.assertIn('source', recommendations[0])
        
    def test_realtime_integration(self):
        """Test realtime integration"""
        # Register a user
        self.integration.register_user(2, 'socket-id-456')
        
        # Check if user is connected
        self.assertTrue(self.integration.is_user_connected(2))
        
        # Record an event through the integration
        success, _, _ = self.integration.record_event(2, 101, 'view')
        self.assertTrue(success)
        
        # Check if an event was emitted
        self.assertGreaterEqual(len(self.mock_socketio.emitted_messages), 1)
        
        # Unregister user
        self.integration.unregister_user(2)
        self.assertFalse(self.integration.is_user_connected(2))
        
    def test_clean_expired_events(self):
        """Test cleaning expired events"""
        # Add event with past timestamp
        past_time = datetime(2020, 1, 1)
        self.recommender.record_event(self.user_id, 101, 'view', timestamp=past_time)
        
        # Clean expired events
        self.recommender._clean_expired_events()
        
        # Check that no events from 2020 remain
        for event in self.recommender.recent_events[self.user_id]:
            self.assertNotEqual(event['timestamp'].year, 2020)


if __name__ == '__main__':
    unittest.main()
