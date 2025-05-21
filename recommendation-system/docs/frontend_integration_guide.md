# Frontend Integration Guide: Travel App Event Tracking & Recommendations

This guide explains how to integrate your frontend travel application with the recommendation system. The recommendation system uses event tracking to generate personalized suggestions for users.

## Installation

1. Copy the `RecommendationList.tsx` component to your components folder
2. Copy the `recommendation.js` utilities to your constants folder or integrate the functions into your existing config file

## Basic Usage

### 1. Add the Recommendation Component to Home Screen

```jsx
import React from 'react';
import { View, ScrollView } from 'react-native';
import RecommendationList from '../components/recommendations/RecommendationList';
import { useUser } from '../context/UserContext';

const HomeScreen = () => {
  const { user } = useUser();
  
  return (
    <ScrollView>
      {/* Other components */}
      
      {/* Recommendations section */}
      <RecommendationList userId={user.id} />
      
      {/* More components */}
    </ScrollView>
  );
};

export default HomeScreen;
```

### 2. Add Recommendation Component to Location Detail Screen

```jsx
import React, { useEffect } from 'react';
import { View, ScrollView } from 'react-native';
import RecommendationList from '../components/recommendations/RecommendationList';
import { useUser } from '../context/UserContext';
import { trackEvents } from '../constants/recommendation';

const LocationDetailScreen = ({ route }) => {
  const { locationId } = route.params;
  const { user } = useUser();
  
  // Track view event when screen loads
  useEffect(() => {
    trackEvents.view(user.id, locationId, { screen: 'detail' });
  }, [locationId, user.id]);
  
  return (
    <ScrollView>
      {/* Location details components */}
      
      {/* Similar locations based on current one */}
      <RecommendationList 
        userId={user.id} 
        currentLocationId={locationId}
      />
    </ScrollView>
  );
};

export default LocationDetailScreen;
```

## Tracking User Events

To provide personalized recommendations, the system needs to track user interactions. Use the utility functions to track these events:

### View Events

Track when a user views a location:

```js
import { trackEvents } from '../constants/recommendation';

// In your component
trackEvents.view(userId, locationId, { source: 'search_results' });
```

### Click Events

Track when a user clicks on location elements:

```js
trackEvents.click(userId, locationId, { element: 'book_now_button' });
```

### Booking Events

Track when a user makes a booking:

```js
trackEvents.book(userId, locationId, { 
  booking_id: 'bk123456',
  check_in: '2025-05-25',
  check_out: '2025-05-28',
  number_of_guests: 2
});
```

### Rating Events

Track when a user rates a location:

```js
trackEvents.rate(userId, locationId, 4.5, { comment_added: true });
```

### Search Events

Track when a user searches for locations:

```js
trackEvents.search(userId, 'beach hotels in Bali');
```

### Favorite Events

Track when a user favorites a location:

```js
// When favoriting
trackEvents.favorite(userId, locationId, true);

// When unfavoriting
trackEvents.favorite(userId, locationId, false);
```

## Direct API Access

For more advanced usage, you can access the recommendation API directly:

```js
import { saveEvent, getRecommendations } from '../constants/recommendation';

// Custom event tracking
const trackCustomEvent = async () => {
  const result = await saveEvent({
    user_id: userId,
    location_id: locationId,
    event_type: 'custom_event',
    data: {
      custom_field: 'value'
    }
  });
  
  if (result.success) {
    console.log('Event tracked successfully');
  }
};

// Get recommendations manually
const fetchCustomRecommendations = async () => {
  const result = await getRecommendations(userId, locationId, 'content_based');
  
  if (result.success && result.recommendations) {
    setLocations(result.recommendations);
  }
};
```

## WebSocket Integration (Advanced)

The recommendation system supports real-time updates via WebSocket. To use this feature:

```jsx
import React, { useEffect, useState } from 'react';
import { View } from 'react-native';
import io from 'socket.io-client';
import { API_URL } from '../constants/config';

const RealtimeRecommendations = ({ userId }) => {
  const [recommendations, setRecommendations] = useState([]);
  
  useEffect(() => {
    // Connect to WebSocket
    const socket = io(API_URL);
    
    // Register user with Socket.IO
    socket.on('connect', () => {
      console.log('Socket connected');
      socket.emit('register_user', { user_id: userId });
    });
    
    // Listen for successful registration
    socket.on('registration_success', (data) => {
      console.log('Registration successful:', data.message);
    });
    
    // Listen for real-time recommendations
    socket.on('realtime_recommendation', (data) => {
      if (data.user_id === userId) {
        console.log('Received real-time recommendations');
        setRecommendations(data.recommendations);
      }
    });
    
    // Clean up socket on unmount
    return () => {
      socket.disconnect();
    };
  }, [userId]);
  
  // Render recommendations...
};
```

## Best Practices

1. **Track Events Consistently**: Make sure to track user events consistently across your app to improve recommendation quality.

2. **Handle Errors Gracefully**: The recommendation system might not always be available. Make sure to handle error cases.

3. **Use Correct User IDs**: Always use the same user ID format across your app.

4. **Test with Different Users**: Create multiple test users with different preferences to verify that recommendations are personalized.

5. **Performance**: The recommendation component includes lazy loading to minimize impact on app performance.

## Troubleshooting

1. **No Recommendations**: New users might not receive recommendations immediately. The system needs some interaction data to generate personalized suggestions.

2. **WebSocket Connection Issues**: Make sure your server URL is correct and the server is running.

3. **Event Tracking Not Working**: Check the network tab to ensure events are being sent correctly.

4. **Slow Loading**: If recommendations are slow to load, consider pre-fetching them when the app starts.

For further assistance, contact the backend team.
