"""
Utility script to run the real-time recommendation system with sample data
"""
import os
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_demo_data():
    """
    Setup sample data for demonstration
    """
    print("Setting up demo data...")
    
    # Create directories if they don't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create sample product data
    products = []
    categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
    for i in range(1, 101):
        category = random.choice(categories)
        price = round(random.uniform(10, 500), 2)
        rating = round(random.uniform(1, 5), 1)
        
        products.append({
            "product_id": i,
            "name": f"Product {i}",
            "description": f"This is product {i} in the {category} category",
            "category": category,
            "price": price,
            "rating": rating
        })
    
    products_df = pd.DataFrame(products)
    
    # Create sample ratings data
    ratings = []
    for user_id in range(1, 21):  # 20 users
        # Each user rates 5-15 products
        num_ratings = random.randint(5, 15)
        rated_products = random.sample(range(1, 101), num_ratings)
        
        for product_id in rated_products:
            rating = random.randint(1, 5)
            ratings.append({
                "user_id": user_id,
                "product_id": product_id,
                "rating": rating,
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")
            })
    
    ratings_df = pd.DataFrame(ratings)
    
    # Create sample events data
    events = []
    event_types = ["view", "add_to_cart", "purchase"]
    weights = [0.7, 0.2, 0.1]  # More views than purchases
    
    for user_id in range(1, 21):  # 20 users
        # Each user has 10-30 events
        num_events = random.randint(10, 30)
        for _ in range(num_events):
            product_id = random.randint(1, 100)
            event_type = random.choices(event_types, weights=weights, k=1)[0]
            
            # More recent timestamps
            hours_ago = random.randint(0, 48)
            timestamp = (datetime.now() - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
            
            events.append({
                "user_id": user_id,
                "product_id": product_id,
                "event_type": event_type,
                "timestamp": timestamp
            })
    
    events_df = pd.DataFrame(events)
    
    # Save the data
    products_df.to_csv(os.path.join(raw_dir, "products.csv"), index=False)
    ratings_df.to_csv(os.path.join(raw_dir, "dataset.csv"), index=False)
    events_df.to_csv(os.path.join(raw_dir, "events.csv"), index=False)
    
    # Save integrated data
    integrated_df = pd.merge(ratings_df, products_df, on="product_id")
    integrated_df.to_csv(os.path.join(processed_dir, "integrated_data.csv"), index=False)
    
    print(f"Created {len(products_df)} products")
    print(f"Created {len(ratings_df)} ratings")
    print(f"Created {len(events_df)} events")
    print("Demo data setup complete!")


def run_app():
    """Run the recommendation system app"""
    try:
        # Check if we're running from the recommendation-system directory or the parent
        if os.path.basename(os.getcwd()) == "recommendation-system":
            # Already in recommendation-system directory
            import main as app_module
        else:
            # In parent directory, need to import from recommendation-system
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            "recommendation-system"))
            try:
                import main as app_module
            except ImportError:
                print("Could not import main from recommendation-system directory.")
                print("Trying to import from current directory...")
                import main as app_module
        
        print("Starting the recommendation system with real-time capabilities...")
        print("Access the demo at: http://localhost:5000/demo/realtime")
        print("Press Ctrl+C to stop")
        
        # Get the app and socketio objects
        app = getattr(app_module, "app", None)
        socketio = getattr(app_module, "socketio", None)
        
        if app and socketio:
            # Use socketio if available
            socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
        elif app:
            # Fall back to regular Flask server if only app is available
            print("SocketIO not found, using regular Flask server (real-time features will not work)")
            app.run(debug=True, port=5000)
        else:
            print("Error: Could not find app object in main module.")
    except ImportError as e:
        print(f"Error: Could not import the main application: {e}")
        print("Please make sure you're running this script from the project root directory.")


if __name__ == "__main__":
    # Check if we need to setup demo data
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_demo_data()
    else:
        run_app()
