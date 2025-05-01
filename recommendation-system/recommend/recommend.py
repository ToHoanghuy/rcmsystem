import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from recommenders.collaborative import collaborative_filtering
from recommenders.content_based import recommend_content_based
from recommenders.event_based import recommend_based_on_events, recommend_based_on_events_advanced
from recommenders.hybrid import hybrid_recommendations, weighted_hybrid, adaptive_hybrid
from config.config import get_config
from models.evaluation import evaluate_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(config=None):
    """
    Load necessary data files for recommendations
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary with data paths
        
    Returns:
    --------
    tuple
        Tuple containing (ratings_df, product_details_df, events_df)
    """
    if config is None:
        config = get_config()
    
    data_dir = config.get('data_dir', 'data')
    
    # Load ratings data
    ratings_path = os.path.join(data_dir, 'processed', config.get('ratings_file', 'intergrated_data.csv'))
    try:
        ratings_df = pd.read_csv(ratings_path)
        logger.info(f"Loaded ratings data with {len(ratings_df)} rows")
    except Exception as e:
        logger.error(f"Error loading ratings data: {str(e)}")
        ratings_df = pd.DataFrame()
    
    # Load product details
    products_path = os.path.join(data_dir, 'raw', config.get('products_file', 'products.csv'))
    try:
        product_details_df = pd.read_csv(products_path)
        logger.info(f"Loaded product details with {len(product_details_df)} rows")
    except Exception as e:
        logger.error(f"Error loading product details: {str(e)}")
        product_details_df = pd.DataFrame()
    
    # Load events data if available
    events_df = pd.DataFrame()
    events_path = os.path.join(data_dir, 'raw', config.get('events_file', 'events.csv'))
    try:
        if os.path.exists(events_path):
            events_df = pd.read_csv(events_path)
            logger.info(f"Loaded events data with {len(events_df)} rows")
    except Exception as e:
        logger.error(f"Error loading events data: {str(e)}")
    
    return ratings_df, product_details_df, events_df

def generate_recommendations(model_type, user_id, data=None, product_details=None, events_data=None, 
                           num_recommendations=10, case='default', product_id=None, weights=None):
    """
    Generate recommendations based on the specified model type and case
    
    Parameters:
    -----------
    model_type : str
        Type of recommendation model to use ('collaborative', 'content', 'event', 'hybrid', 'adaptive')
    user_id : int
        User ID for which to generate recommendations
    data : DataFrame, optional
        DataFrame containing ratings data
    product_details : DataFrame, optional
        DataFrame containing product details
    events_data : DataFrame, optional
        DataFrame containing events data
    num_recommendations : int, optional
        Number of recommendations to return
    case : str, optional
        Special case for recommendations ('default', 'popular', 'related', 'user_history')
    product_id : int, optional
        Product ID for related product recommendations
    weights : dict, optional
        Dictionary of weights for hybrid recommendations
        
    Returns:
    --------
    list
        List of dictionaries containing recommended products
    """
    # Load data if not provided
    if data is None or product_details is None:
        ratings_df, product_details_df, events_df = load_data()
        data = data if data is not None else ratings_df
        product_details = product_details if product_details is not None else product_details_df
        events_data = events_data if events_data is not None else events_df
    
    # Handle special cases first
    if case == 'popular':
        # Return popular products based on ratings
        if 'rating' in product_details.columns:
            popular_products = product_details.sort_values(by='rating', ascending=False).head(num_recommendations)
        else:
            # If no ratings column, try to infer popularity from data
            if not data.empty and 'product_id' in data.columns and 'rating' in data.columns:
                product_avg_ratings = data.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
                # Consider both rating and number of ratings for popularity
                product_avg_ratings['popularity_score'] = product_avg_ratings['mean'] * np.log1p(product_avg_ratings['count'])
                popular_product_ids = product_avg_ratings.sort_values('popularity_score', ascending=False).head(num_recommendations)['product_id']
                popular_products = product_details[product_details['product_id'].isin(popular_product_ids)]
            else:
                popular_products = product_details.head(num_recommendations)
        
        return popular_products.to_dict(orient='records')
    
    elif case == 'related' and product_id is not None:
        # Find related products based on similarity
        try:
            # Select relevant features for similarity calculation
            feature_cols = []
            for col in ['product_type', 'category', 'location', 'price', 'rating']:
                if col in product_details.columns:
                    feature_cols.append(col)
            
            if not feature_cols:
                logger.warning("No suitable feature columns found for similarity calculation")
                return []
            
            product_features = product_details[feature_cols]
            
            # Handle different data types appropriately
            numeric_features = product_features.select_dtypes(include=['number']).columns
            categorical_features = product_features.select_dtypes(exclude=['number']).columns
            
            # Process numerical features
            if not numeric_features.empty:
                # Normalize numerical features
                numeric_data = product_features[numeric_features].copy()
                for col in numeric_features:
                    if numeric_data[col].std() > 0:
                        numeric_data[col] = (numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std()
            else:
                numeric_data = pd.DataFrame(index=product_features.index)
            
            # Process categorical features
            if not categorical_features.empty:
                categorical_data = pd.get_dummies(product_features[categorical_features])
            else:
                categorical_data = pd.DataFrame(index=product_features.index)
            
            # Combine features
            processed_features = pd.concat([numeric_data, categorical_data], axis=1)
            
            # Calculate similarity
            similarity_matrix = cosine_similarity(processed_features)
            product_indices = product_details.reset_index().set_index('product_id')
            
            # Find index of the target product
            product_idx = None
            for i, pid in enumerate(product_details['product_id']):
                if pid == product_id:
                    product_idx = i
                    break
            
            if product_idx is not None:
                # Get similar products indices
                similar_indices = similarity_matrix[product_idx].argsort()[-(num_recommendations+1):][::-1]
                # Remove the product itself if it's in the results
                similar_indices = [idx for idx in similar_indices if product_details.iloc[idx]['product_id'] != product_id]
                # Get the products
                related_products = product_details.iloc[similar_indices[:num_recommendations]]
                return related_products.to_dict(orient='records')
            else:
                logger.warning(f"Product ID {product_id} not found in product details")
                return []
        except Exception as e:
            logger.error(f"Error finding related products: {str(e)}")
            return []
    
    elif case == 'user_history' and user_id is not None:
        # Return user's past interactions
        user_history = data[data['user_id'] == user_id]
        if user_history.empty:
            return []
            
        # Combine with events if available
        if events_data is not None and not events_data.empty:
            user_events = events_data[events_data['user_id'] == user_id]
            if not user_events.empty:
                # Add products from events to history
                event_products = user_events['product_id'].unique()
                for pid in event_products:
                    if pid not in user_history['product_id'].values:
                        # Create a dummy rating entry based on event
                        if 'event_type' in user_events.columns:
                            # Use most significant event for this product
                            product_events = user_events[user_events['product_id'] == pid]
                            event_type = product_events['event_type'].iloc[0]
                            implicit_rating = 3.0  # Default
                            # Assign implicit rating based on event type
                            if event_type == 'purchase':
                                implicit_rating = 5.0
                            elif event_type == 'add_to_cart':
                                implicit_rating = 4.0
                            elif event_type == 'detail_product_view':
                                implicit_rating = 3.5
                            user_history = pd.concat([user_history, pd.DataFrame({
                                'user_id': [user_id],
                                'product_id': [pid],
                                'rating': [implicit_rating],
                                'from_event': [True]
                            })])
        
        # Sort by rating and return
        top_products = user_history.sort_values(by='rating', ascending=False).head(num_recommendations)
        history_products = product_details[product_details['product_id'].isin(top_products['product_id'])]
        if not history_products.empty:
            # Add rating from user history to output
            history_products = pd.merge(
                history_products, 
                top_products[['product_id', 'rating']], 
                on='product_id',
                how='left'
            )
            return history_products.to_dict(orient='records')
        return []
    
    # Regular recommendation generation based on model type
    recommendations = pd.DataFrame()
    
    try:
        if model_type == 'collaborative':
            # Collaborative filtering recommendations
            recommendations = collaborative_filtering(
                user_id=user_id,
                ratings_df=data,
                product_df=product_details,
                top_n=num_recommendations
            )
        
        elif model_type == 'content':
            # Content-based recommendations
            recommendations = recommend_content_based(
                user_id=user_id,
                ratings=data[data['user_id'] == user_id] if not data.empty else None,
                product_details=product_details,
                top_n=num_recommendations
            )
        
        elif model_type == 'event':
            # Event-based recommendations
            if events_data is not None and not events_data.empty:
                try:
                    # First try advanced event-based recommendations
                    recommendations = recommend_based_on_events_advanced(
                        user_id=user_id,
                        events_path=events_data,
                        product_details=product_details,
                        top_n=num_recommendations
                    )
                except Exception as e:
                    logger.warning(f"Advanced event-based recommendations failed: {str(e)}")
                    # Fall back to basic event-based recommendations
                    recommendations = recommend_based_on_events(
                        user_id=user_id,
                        events_path=events_data,
                        product_details=product_details,
                        top_n=num_recommendations
                    )
            else:
                logger.warning("No event data available for event-based recommendations")
        
        elif model_type == 'hybrid':
            # Hybrid recommendations
            if weights is not None:
                recommendations = weighted_hybrid(
                    user_id=user_id,
                    ratings_df=data,
                    product_df=product_details,
                    events_df=events_data,
                    weights=weights,
                    top_n=num_recommendations
                )
            else:
                recommendations = hybrid_recommendations(
                    user_id=user_id,
                    ratings_df=data,
                    product_df=product_details,
                    events_df=events_data,
                    top_n=num_recommendations
                )
        
        elif model_type == 'adaptive':
            # Adaptive hybrid recommendations
            recommendations = adaptive_hybrid(
                user_id=user_id,
                ratings_df=data,
                product_df=product_details,
                events_df=events_data,
                top_n=num_recommendations
            )
        
        else:
            logger.warning(f"Unknown model type: {model_type}")
            # Provide popular products as fallback
            return generate_recommendations(None, user_id, data, product_details, events_data, 
                                         num_recommendations, case='popular')
    
    except Exception as e:
        logger.error(f"Error generating {model_type} recommendations: {str(e)}")
        # Provide popular products as fallback
        return generate_recommendations(None, user_id, data, product_details, events_data, 
                                     num_recommendations, case='popular')
    
    # Check if recommendations were successfully generated
    if recommendations is None or recommendations.empty:
        logger.warning(f"No {model_type} recommendations generated for user {user_id}")
        # Provide popular products as fallback
        return generate_recommendations(None, user_id, data, product_details, events_data, 
                                     num_recommendations, case='popular')
    
    return recommendations.to_dict(orient='records')

def get_recommendations(user_id, model_type='hybrid', num_recommendations=10, include_reasons=False):
    """
    High-level function to get recommendations for a user
    
    Parameters:
    -----------
    user_id : int
        User ID for which to generate recommendations
    model_type : str, optional
        Type of recommendation model to use
    num_recommendations : int, optional
        Number of recommendations to return
    include_reasons : bool, optional
        Whether to include explanation for recommendations
        
    Returns:
    --------
    dict
        Dictionary containing recommendations and optional metadata
    """
    # Load data
    ratings_df, product_details_df, events_df = load_data()
    
    # Generate recommendations
    recommended_products = generate_recommendations(
        model_type=model_type,
        user_id=user_id,
        data=ratings_df,
        product_details=product_details_df,
        events_data=events_df,
        num_recommendations=num_recommendations
    )
    
    result = {
        'user_id': user_id,
        'model_type': model_type,
        'recommendations': recommended_products
    }
    
    # Add explanation reasons if requested
    if include_reasons and recommended_products:
        reasons = []
        
        for product in recommended_products:
            product_id = product.get('product_id')
            product_name = product.get('name', f"Product {product_id}")
            
            reason = f"{product_name} is recommended"
            
            if 'sources' in product:
                sources = product['sources'] 
                if 'collaborative' in sources:
                    reason += " because users with similar preferences liked it"
                if 'content' in sources:
                    reason += " because it's similar to products you've liked"
                if 'event' in sources:
                    reason += " based on your browsing behavior"
            
            if 'hybrid_score' in product:
                confidence = min(100, int(product['hybrid_score'] / 5 * 100))
                reason += f" (Confidence: {confidence}%)"
                
            reasons.append({
                'product_id': product_id,
                'reason': reason
            })
        
        result['reasons'] = reasons
    
    return result