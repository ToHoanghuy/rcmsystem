from flask import Flask, request, jsonify
import pandas as pd
from scripts.preprocess import preprocess_data
from scripts.train_model import train_all_models
from scripts.generate_recommendations import generate_recommendations

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đường dẫn đến dữ liệu
RATINGS_PATH = "data/raw/dataset.csv"
PRODUCTS_PATH = "data/raw/products.csv"
EVENTS_PATH = "data/raw/events.csv"

# Xử lý dữ liệu
print("Processing data...")
data = preprocess_data(RATINGS_PATH, PRODUCTS_PATH, EVENTS_PATH)
product_details = pd.read_csv(PRODUCTS_PATH)

# Huấn luyện mô hình
print("Training models...")
collaborative_model, content_similarity = train_all_models(data, product_details)

# API gợi ý
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    case = request.args.get('case', default='hybrid', type=str)
    product_id = request.args.get('product_id', type=int)

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    if case == 'hybrid':
        recommendations = generate_recommendations(user_id, collaborative_model, content_similarity, product_details)
    elif case == 'collaborative':
        recommendations = collaborative_model.predict(user_id)
    elif case == 'content_based' and product_id is not None:
        recommendations = content_similarity[product_id]
    else:
        return jsonify({"error": "Invalid case or missing product_id for content-based recommendations"}), 400

    return jsonify({"user_id": user_id, "recommendations": recommendations})

# Điểm khởi chạy ứng dụng
if __name__ == "__main__":
    app.run(debug=True)