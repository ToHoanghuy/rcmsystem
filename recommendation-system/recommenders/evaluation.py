import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calculate precision and recall at k for recommendations
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
    k : int
        The number of recommendations to consider
    threshold : float
        The rating threshold above which an item is considered relevant
        
    Returns:
    --------
    tuple
        (precision, recall) at k
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating in descending order
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Number of relevant items for this user
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        # Number of recommended items at k
        n_rec_k = min(k, len(user_ratings))
        
        # Number of relevant and recommended items at k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Calculate precision and recall
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1.0
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate mean precision and recall
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return mean_precision, mean_recall

def ndcg_at_k(predictions, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain at k
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
    k : int
        The number of recommendations to consider
        
    Returns:
    --------
    float
        NDCG@k score
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    ndcg_scores = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating (prediction)
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        est_ranks = [r for _, r in user_ratings[:k]]
        
        # Sort by true rating
        ideal_ranks = sorted([r for _, r in user_ratings], reverse=True)[:k]
        
        # Calculate DCG
        dcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(est_ranks)])
        
        # Calculate ideal DCG
        idcg = sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_ranks)])
        
        # Calculate NDCG
        ndcg_score = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg_score)
    
    # Mean NDCG across all users
    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return mean_ndcg

def calculate_mae(predictions):
    """
    Calculate Mean Absolute Error for predictions
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
        
    Returns:
    --------
    float
        Mean Absolute Error
    """
    true_ratings = [true_r for _, _, true_r, _, _ in predictions]
    est_ratings = [est for _, _, _, est, _ in predictions]
    
    if not true_ratings or not est_ratings:
        return float('nan')
    
    return mean_absolute_error(true_ratings, est_ratings)

def calculate_rmse(predictions):
    """
    Calculate Root Mean Squared Error for predictions
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
        
    Returns:
    --------
    float
        Root Mean Squared Error
    """
    true_ratings = [true_r for _, _, true_r, _, _ in predictions]
    est_ratings = [est for _, _, _, est, _ in predictions]
    
    if not true_ratings or not est_ratings:
        return float('nan')
    
    return np.sqrt(mean_squared_error(true_ratings, est_ratings))

def map_at_k(predictions, k=10, threshold=3.5):
    """
    Calculate Mean Average Precision at k
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
    k : int
        The number of recommendations to consider
    threshold : float
        The rating threshold above which an item is considered relevant
        
    Returns:
    --------
    float
        MAP@k score
    """
    user_est_true = {}
    for uid, _, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((est, true_r))
    
    aps = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Limit to k recommendations
        user_ratings = user_ratings[:k]
        
        # Check if there are relevant items
        relevant_items = [(i, r) for i, (_, r) in enumerate(user_ratings) if r >= threshold]
        
        if not relevant_items:
            continue
            
        ap = 0.0
        num_correct = 0
        
        for i, r in relevant_items:
            num_correct += 1
            precision_at_i = num_correct / (i + 1)
            ap += precision_at_i
            
        ap /= len(relevant_items)
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall values
    
    Parameters:
    -----------
    precision : float
        Precision value
    recall : float
        Recall value
        
    Returns:
    --------
    float
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_serendipity(predictions, user_profiles, k=10, threshold=3.5):
    """
    Calculate serendipity metric for recommendations
    Serendipity measures how surprising and relevant the recommendations are
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
    user_profiles : dict
        Dictionary mapping user_id to their profile (items they have rated or interacted with)
    k : int
        The number of recommendations to consider
    threshold : float
        The rating threshold above which an item is considered relevant
        
    Returns:
    --------
    float
        Serendipity score
    """
    user_est_true = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((iid, est, true_r))
    
    serendipity_scores = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Get user profile items
        user_items = user_profiles.get(uid, set())
        
        # Calculate unexpectedness for top-k recommendations
        unexpectedness = 0
        valid_items = 0
        
        for i, (iid, est, true_r) in enumerate(user_ratings[:k]):
            if true_r >= threshold:  # Item is relevant
                # Calculate unexpectedness based on distance from user's existing profile
                if iid not in user_items:
                    unexpectedness += 1
                valid_items += 1
        
        # Calculate serendipity for this user
        user_serendipity = unexpectedness / valid_items if valid_items > 0 else 0.0
        serendipity_scores.append(user_serendipity)
    
    # Mean serendipity across all users
    mean_serendipity = np.mean(serendipity_scores) if serendipity_scores else 0.0
    
    return mean_serendipity

def evaluate_recommendations(predictions, k=10, user_profiles=None):
    """
    Evaluate recommendations using multiple metrics
    
    Parameters:
    -----------
    predictions : list
        A list of tuples in the format (user_id, item_id, true_rating, estimated_rating, details)
    k : int
        The number of recommendations to consider for ranking metrics
    user_profiles : dict, optional
        Dictionary mapping user_id to their profile (items they have rated or interacted with)
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Check if predictions list is empty or invalid
    if not predictions:
        logger.warning("Empty predictions list provided for evaluation")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'ndcg': 0.0,
            'map': 0.0,
            'mae': float('nan'),
            'rmse': float('nan'),
            'serendipity': 0.0
        }
    
    # Calculate precision and recall
    try:
        precision, recall = precision_recall_at_k(predictions, k)
    except Exception as e:
        logger.error(f"Error calculating precision/recall: {str(e)}")
        precision, recall = 0.0, 0.0
    
    # Calculate F1 score
    try:
        f1 = calculate_f1_score(precision, recall)
    except Exception as e:
        logger.error(f"Error calculating F1 score: {str(e)}")
        f1 = 0.0
    
    # Calculate NDCG
    try:
        ndcg = ndcg_at_k(predictions, k)
    except Exception as e:
        logger.error(f"Error calculating NDCG: {str(e)}")
        ndcg = 0.0
    
    # Calculate MAP
    try:
        map_score = map_at_k(predictions, k)
    except Exception as e:
        logger.error(f"Error calculating MAP: {str(e)}")
        map_score = 0.0
    
    # Calculate error metrics
    try:
        mae = calculate_mae(predictions)
        rmse = calculate_rmse(predictions)
    except Exception as e:
        logger.error(f"Error calculating error metrics: {str(e)}")
        mae = float('nan')
        rmse = float('nan')
    
    # Calculate serendipity if user profiles are provided
    serendipity = 0.0
    if user_profiles:
        try:
            serendipity = calculate_serendipity(predictions, user_profiles, k)
        except Exception as e:
            logger.error(f"Error calculating serendipity: {str(e)}")
    
    # Return all metrics in a dictionary
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ndcg': ndcg,
        'map': map_score,
        'mae': mae,
        'rmse': rmse,
        'serendipity': serendipity
    }
    
    logger.info(f"Evaluation metrics at k={k}: {metrics}")
    return metrics

def evaluate_event_based(user_items, recommendations, k=10):
    """
    Evaluate event-based recommendations where explicit ratings may not be available
    
    Parameters:
    -----------
    user_items : DataFrame
        DataFrame containing user-item interactions with columns [user_id, product_id, event_score]
    recommendations : dict
        Dictionary mapping user_id to list of recommended product_ids
    k : int
        Number of recommendations to evaluate
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Check if inputs are valid
    if user_items is None or recommendations is None or len(recommendations) == 0:
        logger.warning("Invalid inputs for event-based evaluation")
        return {
            'hit_rate': 0.0,
            'coverage': 0.0,
            'diversity': 0.0
        }
    
    # Evaluate hit rate (fraction of users with at least one recommended item they interacted with)
    hit_count = 0
    
    # All unique items in the recommendations
    all_recommended_items = set()
    
    # Item pair count for diversity metric
    item_pairs = 0
    unique_item_pairs = 0
    
    for user_id, rec_items in recommendations.items():
        # Get items this user has interacted with
        user_data = user_items[user_items['user_id'] == user_id]
        if user_data.empty:
            continue
            
        interacted_items = set(user_data['product_id'].values)
        rec_items_set = set(rec_items[:k])
        
        # Add to all recommended items set
        all_recommended_items.update(rec_items_set)
        
        # Count hits for this user
        if len(rec_items_set.intersection(interacted_items)) > 0:
            hit_count += 1
        
        # Calculate diversity (as uniqueness of item pairs)
        for i, item1 in enumerate(rec_items[:k]):
            for j, item2 in enumerate(rec_items[:k]):
                if i < j:  # Only count unique pairs
                    item_pairs += 1
                    if item1 != item2:
                        unique_item_pairs += 1
    
    # Hit rate
    hit_rate = hit_count / len(recommendations) if len(recommendations) > 0 else 0.0
    
    # Coverage (fraction of all items that were recommended at least once)
    all_items = set(user_items['product_id'].unique())
    coverage = len(all_recommended_items) / len(all_items) if len(all_items) > 0 else 0.0
    
    # Diversity (uniqueness of item pairs)
    diversity = unique_item_pairs / item_pairs if item_pairs > 0 else 1.0
    
    metrics = {
        'hit_rate': hit_rate,
        'coverage': coverage,
        'diversity': diversity
    }
    
    logger.info(f"Event-based evaluation metrics at k={k}: {metrics}")
    return metrics

def compare_recommendation_models(test_data, models_predictions, k=10):
    """
    Compare different recommendation models side by side
    
    Parameters:
    -----------
    test_data : DataFrame
        DataFrame containing test data
    models_predictions : dict
        Dictionary mapping model names to prediction lists
    k : int
        Number of recommendations to evaluate
        
    Returns:
    --------
    DataFrame
        DataFrame containing comparison of evaluation metrics for different models
    """
    results = {}
    
    for model_name, predictions in models_predictions.items():
        try:
            metrics = evaluate_recommendations(predictions, k)
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            results[model_name] = {
                'precision': float('nan'),
                'recall': float('nan'),
                'ndcg': float('nan'),
                'map': float('nan'),
                'mae': float('nan'),
                'rmse': float('nan')
            }
    
    # Convert results to DataFrame for easy comparison
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    logger.info(f"Model comparison at k={k}:\n{results_df}")
    return results_df

def visualize_evaluation_results(results_df, save_path=None):
    """
    Visualize evaluation metrics for different recommendation models
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame containing evaluation metrics for different models
    save_path : str, optional
        Path to save the visualization
    
    Returns:
    --------
    None
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # Create subplots for different metric groups
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy metrics (RMSE, MAE)
        error_metrics = results_df[['rmse', 'mae']]
        error_metrics.plot(kind='bar', ax=axes[0, 0], title='Error Metrics (Lower is Better)')
        axes[0, 0].set_ylabel('Error Value')
        
        # Ranking metrics (NDCG, MAP)
        ranking_metrics = results_df[['ndcg', 'map']]
        ranking_metrics.plot(kind='bar', ax=axes[0, 1], title='Ranking Metrics')
        axes[0, 1].set_ylabel('Score')
        
        # Relevance metrics (Precision, Recall, F1)
        if 'f1' in results_df.columns:
            relevance_metrics = results_df[['precision', 'recall', 'f1']]
        else:
            relevance_metrics = results_df[['precision', 'recall']]
        relevance_metrics.plot(kind='bar', ax=axes[1, 0], title='Relevance Metrics')
        axes[1, 0].set_ylabel('Score')
        
        # Other metrics (Serendipity, Coverage, Diversity)
        other_metrics = pd.DataFrame()
        for col in ['serendipity', 'coverage', 'diversity']:
            if col in results_df.columns:
                other_metrics[col] = results_df[col]
        
        if not other_metrics.empty:
            other_metrics.plot(kind='bar', ax=axes[1, 1], title='Additional Metrics')
            axes[1, 1].set_ylabel('Score')
        else:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Evaluation visualization saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        logger.warning("Matplotlib or seaborn not installed. Skipping visualization.")
    except Exception as e:
        logger.error(f"Error visualizing evaluation results: {str(e)}")