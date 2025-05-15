# Import các module cơ bản
from .collaborative import train_collaborative_model, train_advanced_collaborative_model
from .content_based import train_content_based_model, get_similar_products
from .hybrid import hybrid_recommendations, adaptive_hybrid_recommendations, switching_hybrid_recommendations
from .event_based import recommend_based_on_events, recommend_based_on_events_advanced, analyze_user_behavior_sequence
from .realtime import RealtimeRecommender

# Singleton instance cho RealtimeRecommender để sử dụng trên toàn ứng dụng
realtime_recommender = RealtimeRecommender()