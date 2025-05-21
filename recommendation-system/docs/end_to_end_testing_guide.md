# End-to-End Testing Guide: Travel App Recommendation System

This guide provides instructions for end-to-end testing of the recommendation system integration with the travel application. Follow these steps to verify that the system is working correctly.

## Prerequisites

1. Python server is running (`python main.py` from the recommendation-system directory)
2. MongoDB is installed and running (optional, for event storage testing)
3. Node.js and npm are installed (for frontend testing)

## 1. WebSocket Test Client

The recommendation system includes a built-in WebSocket test client that allows you to simulate user events and see real-time recommendations.

1. Open your browser and navigate to: `http://localhost:5000/websocket-test`
2. Enter a user ID (e.g., "1") in the User ID field
3. Click "Connect" to establish a WebSocket connection
4. Enter a location ID (e.g., "101") in the Location ID field
5. Select an event type (e.g., "view") from the dropdown
6. Click "Send Event" to send the event to the recommendation system
7. You should see real-time recommendations appear in the Recommendations panel
8. Try different event types and observe the changes in recommendations

### What to verify:
- Connection status shows "Connected"
- Events are successfully recorded (check the log panel)
- Recommendations appear after sending events
- Different event types produce different recommendations

## 2. API Integration Test

Test the REST API endpoints directly to ensure they're responding correctly.

```bash
# Send a view event
curl -X POST http://localhost:5000/api/track \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_1", "location_id": "101", "event_type": "view", "data": {"source": "test"}}'

# Get recommendations for a user
curl http://localhost:5000/api/recommend?user_id=test_user_1&case=hybrid

# Test content-based recommendations for a specific location
curl http://localhost:5000/api/recommend?user_id=test_user_1&case=content_based&location_id=101
```

### What to verify:
- API responds with 200 status code
- Response includes recommendations array
- The system correctly identifies if recommendations were sent via WebSocket

## 3. WebSocket Integration Test

Run the WebSocket integration test script to simulate multiple users connecting and generating events simultaneously:

```bash
# Navigate to the test directory
cd python/recommendation-system/test

# Run the WebSocket test
python test_websocket_client.py
```

### What to verify:
- Multiple users can connect simultaneously
- Events are processed correctly
- Recommendations are received for all connected users
- No errors or exceptions are reported

## 4. Performance Testing

Run the performance test to measure the system's response time and throughput under various load conditions:

```bash
# Navigate to the test directory
cd python/recommendation-system/test

# Run the performance test
python performance_test.py
```

### What to verify:
- Response times remain acceptable under load
- Error rates remain low (ideally zero)
- Throughput scales with concurrency
- Check the generated performance_results_*.png file for visual analysis

## 5. Frontend Integration Test

Test the integration with the actual travel app frontend:

1. Start the React Native development server:
   ```bash
   cd SE121_TravelSocial
   npm start
   ```

2. Open the app in your emulator or device
3. Navigate to a location details page
4. Verify that recommendations appear
5. Interact with different locations (view, book, rate)
6. Verify that recommendations update based on your interactions

### What to verify:
- RecommendationList component renders properly
- Events are tracked when interacting with locations
- Recommendations reflect user preferences over time

## 6. MongoDB Storage Test (if configured)

If you've configured MongoDB storage, verify that events and recommendations are being stored correctly:

1. Open MongoDB shell or a GUI tool like MongoDB Compass
2. Connect to your database (default: `travel_recommendations`)
3. Check the `events` collection for stored user events
4. Check the `ratings` collection for user ratings
5. Check the `recommendations` collection for stored recommendations

### What to verify:
- Events have correct user_id, location_id, and event_type
- Timestamps are recorded properly
- Rating values are stored correctly
- Recommendations include all necessary metadata

## 7. Caching Test

Test that the recommendation caching system is working correctly:

1. Clear the server logs
2. Make the same recommendation request twice:
   ```bash
   curl http://localhost:5000/api/recommend?user_id=test_user_1&case=hybrid
   curl http://localhost:5000/api/recommend?user_id=test_user_1&case=hybrid
   ```
3. Check the server logs for cache hit messages

### What to verify:
- First request shows cache miss
- Second request shows cache hit
- Response time for the second request should be significantly faster

## 8. End-to-End User Flow

Finally, test a complete user flow:

1. Connect a new user via the WebSocket test client
2. Send a series of view events for different locations
3. Send a booking event for one location
4. Send a rating event for another location
5. Check the recommendations after each action
6. Verify that the MongoDB database has all events recorded

### What to verify:
- User's recommendations evolve based on their behavior
- Booking and rating events have higher influence than view events
- All events are properly stored in the database
- No errors occur during the entire flow

## Troubleshooting

If you encounter issues during testing:

1. Check the server logs for error messages
2. Verify MongoDB connection if events aren't being stored
3. Check for WebSocket connection issues in the browser console
4. Verify API endpoints are accessible and returning correct data
5. Ensure the recommendation cache is functioning properly

## Reporting Results

After completing the tests, document your findings including:

1. Which tests passed/failed
2. Performance metrics (response time, throughput)
3. Any errors or unexpected behavior
4. Suggestions for improvement

This comprehensive testing ensures that the recommendation system is functioning correctly and ready for production deployment.
