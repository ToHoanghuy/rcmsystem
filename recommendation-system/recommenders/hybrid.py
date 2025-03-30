def hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details, num_recommendations=10):
    collaborative_recommendations = collaborative_model.predict(user_id)
    content_recommendations = content_similarity[user_id]
    combined_scores = collaborative_recommendations + content_recommendations
    top_indices = combined_scores.argsort()[-num_recommendations:][::-1]
    return product_details.iloc[top_indices]