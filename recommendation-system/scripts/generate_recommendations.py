import pandas as pd
import logging
from recommenders.hybrid import hybrid_recommendations, adaptive_hybrid_recommendations as adaptive_hybrid
from recommenders.hybrid import context_aware_hybrid_recommendations, personalized_matrix_factorization_hybrid, switching_hybrid_recommendations

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

def generate_location_recommendations(user_id, collaborative_model, content_model_data, product_details, 
                               user_events=None, user_location=None, top_n=10):
    """
    Generate location-based recommendations with context awareness
    
    Parameters:
    -----------
    user_id : int
        User ID for which recommendations are to be generated
    collaborative_model : object
        Trained collaborative filtering model
    content_model_data : tuple
        Content-based model data (similarity_matrix, index_to_id, id_to_index)
    product_details : DataFrame
        DataFrame containing location details (with province, address, etc.)
    user_events : DataFrame, optional
        DataFrame containing user event data
    user_location : dict, optional
        Dictionary containing user's current location {'latitude': float, 'longitude': float}
    top_n : int, optional
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended locations with scores
    """
    logger.info(f"Generating location-based recommendations for user {user_id}")
    
    try:
        # Create context object with location data if available
        user_context = None
        if user_location:
            user_context = {
                'location': user_location,
                'time_of_day': 'afternoon'  # Could be dynamically set based on current time
            }
            
            # If we have location data for the user, add some location preferences
            # This is a simple example - in reality would be based on user history
            if 'province' in product_details.columns:
                provinces = product_details['province'].unique().tolist()
                if provinces:
                    # Just picking the first few provinces as "preferred" for demonstration
                    user_context['preferred_provinces'] = provinces[:3]
            
            # If we have category data, add some category preferences
            if 'category' in product_details.columns:
                categories = product_details['category'].unique().tolist()
                if categories:
                    user_context['preferred_categories'] = categories[:2]
        
        # Use context-aware hybrid recommendation algorithm
        recommendations = context_aware_hybrid_recommendations(
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            user_id=user_id,
            product_details=product_details,
            user_events=user_events,
            user_context=user_context,
            top_n=top_n
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating location-based recommendations: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def generate_personalized_location_recommendations(user_id, collaborative_model, content_model_data, product_details, 
                                              user_events=None, top_n=10):
    """
    Generate personalized location recommendations using matrix factorization hybrid approach
    
    Parameters:
    -----------
    user_id : int
        User ID for which recommendations are to be generated
    collaborative_model : object
        Trained collaborative filtering model
    content_model_data : tuple
        Content-based model data (similarity_matrix, index_to_id, id_to_index)
    product_details : DataFrame
        DataFrame containing location details
    user_events : DataFrame, optional
        DataFrame containing user event data
    top_n : int, optional
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended locations with scores
    """
    logger.info(f"Generating personalized matrix factorization recommendations for user {user_id}")
    
    try:
        # Use personalized matrix factorization hybrid recommendation algorithm
        recommendations = personalized_matrix_factorization_hybrid(
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            user_id=user_id,
            product_details=product_details,
            user_events=user_events,
            top_n=top_n
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating personalized matrix factorization recommendations: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def generate_switching_location_recommendations(user_id, collaborative_model, content_model_data, product_details, 
                                           confidence_threshold=0.8, top_n=10):
    """
    Generate location recommendations using switching hybrid approach
    
    Parameters:
    -----------
    user_id : int
        User ID for which recommendations are to be generated
    collaborative_model : object
        Trained collaborative filtering model
    content_model_data : tuple
        Content-based model data (similarity_matrix, index_to_id, id_to_index)
    product_details : DataFrame
        DataFrame containing location details
    confidence_threshold : float, optional
        Threshold for switching between collaborative and content-based filtering
    top_n : int, optional
        Number of recommendations to return
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing recommended locations with scores
    string
        Recommendation type used ("collaborative" or "content_based")
    """
    logger.info(f"Generating switching hybrid recommendations for user {user_id}")
    
    try:
        # Use switching hybrid recommendation algorithm
        recommendations, rec_type = switching_hybrid_recommendations(
            collaborative_model=collaborative_model,
            content_model_data=content_model_data,
            user_id=user_id,
            product_details=product_details,
            confidence_threshold=confidence_threshold,
            top_n=top_n
        )
        
        return recommendations, rec_type
        
    except Exception as e:
        logger.error(f"Error generating switching hybrid recommendations: {str(e)}")
        return pd.DataFrame(), "error"  # Return empty DataFrame on error