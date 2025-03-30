from recommenders.hybrid import hybrid_recommendations

def generate_recommendations(user_id, collaborative_model, content_similarity, product_details):
    recommendations = hybrid_recommendations(collaborative_model, content_similarity, user_id, product_details)
    return recommendations