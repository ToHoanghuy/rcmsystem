# Real-time Recommendation System

A complete recommendation system with real-time capabilities, built with Flask and Socket.IO.

## Features

- Collaborative Filtering Recommendations
- Content-Based Filtering Recommendations
- Hybrid Recommendations
- Event-Based Recommendations
- **Real-time Recommendations** with WebSockets

## Requirements

- Python 3.8+
- Flask
- Flask-SocketIO
- Flask-APScheduler (optional, for scheduled cleanup tasks)
- Pandas, NumPy, scikit-learn
- scikit-surprise

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Running the Application

```
python main.py
```

The server will start at http://localhost:5000.

## API Endpoints

### Recommendations

- `GET /recommend` - Get recommendations based on different algorithms
  - Query params:
    - `user_id`: User ID (required except for content-based and popular recommendations)
    - `case`: Type of recommendation (collaborative, content_based, hybrid, behavior_based, popular)
    - `product_id`: Product ID (required for content-based)
    - `method`: Hybrid method (adaptive, switching)

### Real-time Recommendations

- `POST /event` - Record user events and get real-time recommendations
  - Body params (JSON):
    - `user_id`: User ID (required)
    - `product_id`: Product ID (required)
    - `event_type`: Event type (view, add_to_cart, purchase) (required)
    - `timestamp`: Event timestamp (optional)

- `GET /realtime-recommend` - Get real-time recommendations based on user history
  - Query params:
    - `user_id`: User ID (required)
    - `product_id`: Current product ID (optional)
    - `event_type`: Current event type (default: view) (optional)

### Demo

- `GET /demo/realtime` - Real-time recommendation demo page

## WebSocket Events

- `register_user` - Register a user for real-time recommendations
  - Data: `{ user_id: <user_id> }`
- `realtime_recommendation` - Event emitted when new recommendations are available
  - Data: `{ user_id: <user_id>, recommendations: [...] }`

## Architecture

The recommendation system uses multiple algorithms to provide recommendations:

1. **Collaborative Filtering**: Based on user-item interactions
2. **Content-Based Filtering**: Based on item features
3. **Hybrid**: Combination of collaborative and content-based
4. **Event-Based**: Based on user events and behavior
5. **Real-time**: Updates recommendations immediately when users interact with items

## Real-time Feature

The real-time feature uses WebSockets to push recommendations to users as soon as they interact with products. This creates a more dynamic and personalized experience:

- When a user views, adds to cart, or purchases a product, the system immediately analyzes the event
- The system generates fresh recommendations based on the latest action
- Recommendations are pushed to the user instantly via WebSockets
- The system maintains a co-occurrence matrix that tracks which products are viewed or purchased together

## Project Structure

- `main.py`: Main application entry point
- `config/`: Configuration files
- `recommenders/`: Recommendation algorithms
  - `collaborative.py`: Collaborative filtering models
  - `content_based.py`: Content-based filtering models
  - `hybrid.py`: Hybrid recommendation models
  - `event_based.py`: Event-based recommendation models
  - `realtime.py`: Real-time recommendation engine
- `utils/`: Utility functions
  - `realtime_integration.py`: WebSocket integration for real-time recommendations
  - `cache.py`: Caching mechanisms
  - `batch_processing.py`: Batch processing utilities
- `static/`: Static files including the demo page
- `models/`: Model evaluation
- `preprocess/`: Data preprocessing functions
- `test/`: Unit tests
