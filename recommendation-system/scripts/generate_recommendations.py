import pandas as pd
import logging
from recommenders.hybrid import hybrid_recommendations, adaptive_hybrid_recommendations as adaptive_hybrid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_recommendations(user_id, collaborative_model, content_similarity, product_details, user_events=None, top_n=10):
    """
    Generate hybrid recommendations for a user
    
    Parameters:
    -----------
    user_id : int
        User ID for which recommendations are to be generated
    collaborative_model : object
        Trained collaborative filtering model
    content_similarity : object
        Content-based similarity matrix or model
    product_details : DataFrame
        DataFrame containing product details
    user_events : DataFrame, optional
        DataFrame containing user event data
    top_n : int, optional
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended products with scores
    """
    logger.info(f"Generating recommendations for user {user_id}")
    
    try:
        # Convert model data to ratings DataFrame format expected by hybrid_recommendations
        ratings_df = pd.DataFrame()
        
        # Extract user ratings from collaborative model if available
        if hasattr(collaborative_model, 'trainset'):
            for uid, iid, rating in collaborative_model.trainset.all_ratings():
                user = collaborative_model.trainset.to_raw_uid(uid)
                item = collaborative_model.trainset.to_raw_iid(iid)
                row = {'user_id': user, 'product_id': item, 'rating': rating}
                ratings_df = pd.concat([ratings_df, pd.DataFrame([row])], ignore_index=True)
        elif isinstance(collaborative_model, pd.DataFrame):
            # The model is already a ratings DataFrame
            ratings_df = collaborative_model
            
        # Call hybrid_recommendations with the correct parameters
        recommendations = hybrid_recommendations(
            user_id=user_id,
            ratings_df=ratings_df,
            product_df=product_details,
            events_df=user_events,
            top_n=top_n
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def generate_adaptive_recommendations(user_id, collaborative_model, content_similarity, product_details, user_events=None, top_n=10):
    """
    Generate adaptive hybrid recommendations for a user where weights are chosen based on data availability
    
    Parameters:
    -----------
    user_id : int
        User ID for which recommendations are to be generated
    collaborative_model : object
        Trained collaborative filtering model
    content_similarity : object
        Content-based similarity matrix or model
    product_details : DataFrame
        DataFrame containing product details
    user_events : DataFrame, optional
        DataFrame containing user event data
    top_n : int, optional
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended products with scores
    """
    logger.info(f"Generating adaptive recommendations for user {user_id}")
    
    try:
        # Convert model data to ratings DataFrame format expected by hybrid_recommendations
        ratings_df = pd.DataFrame()
        
        # Extract user ratings from collaborative model if available
        if hasattr(collaborative_model, 'trainset'):
            for uid, iid, rating in collaborative_model.trainset.all_ratings():
                user = collaborative_model.trainset.to_raw_uid(uid)
                item = collaborative_model.trainset.to_raw_iid(iid)
                row = {'user_id': user, 'product_id': item, 'rating': rating}
                ratings_df = pd.concat([ratings_df, pd.DataFrame([row])], ignore_index=True)
        elif isinstance(collaborative_model, pd.DataFrame):
            # The model is already a ratings DataFrame
            ratings_df = collaborative_model
            
        # Call adaptive_hybrid with the correct parameters
        recommendations = adaptive_hybrid(
            user_id=user_id,
            ratings_df=ratings_df,
            product_df=product_details,
            events_df=user_events,
            top_n=top_n
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating adaptive hybrid recommendations: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error