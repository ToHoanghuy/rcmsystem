import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pandas as pd
from preprocess import preprocess, load_product_details, load_event_data, integrate_event_data
from recommend import generate_recommendations
from models.model import train_model

app = Flask(__name__)

# Load and preprocess data
try:
    ratings = preprocess(r"D:\python\recommendation-system\data\dataset.csv")
    product_details = load_product_details(r"D:\python\recommendation-system\data\products.csv")
    events = load_event_data(r"D:\python\recommendation-system\data\events.csv")
    data = integrate_event_data(ratings, events)
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Train the recommendation model
model = train_model(data)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    case = request.args.get('case', 'default')
    product_id = request.args.get('product_id')
    
    if user_id is None and case == 'default':
        return jsonify({"error": "User ID is required"}), 400
    
    try:
        user_id = int(user_id) if user_id is not None else None
        product_id = int(product_id) if product_id is not None else None
    except ValueError:
        return jsonify({"error": "User ID and Product ID must be integers"}), 400
    
    recommendations = generate_recommendations(model, user_id, data, product_details, case=case, product_id=product_id)
    return jsonify({"user_id": user_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)