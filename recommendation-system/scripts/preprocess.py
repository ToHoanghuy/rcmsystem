import pandas as pd
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess.preprocess import preprocess_data, load_data, load_event_data, load_product_details, integrate_event_data

def preprocess_pipeline(ratings_df=None, products_df=None, events_df=None, normalize=True):
    """
    Preprocess and integrate data from multiple sources
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame, optional
        DataFrame containing user ratings
    products_df : pandas.DataFrame, optional
        DataFrame containing product details
    events_df : pandas.DataFrame, optional
        DataFrame containing user-product events
    normalize : bool, optional
        Whether to normalize numeric features
    
    Returns:
    --------
    pandas.DataFrame
        Integrated dataset with all available features
    """
    if ratings_df is None and products_df is None and events_df is None:
        # Default behavior: use files from data directory
        root_dir = Path(__file__).parent.parent
        ratings_path = os.path.join(root_dir, 'data', 'raw', 'dataset.csv')
        products_path = os.path.join(root_dir, 'data', 'raw', 'products.csv')
        events_path = os.path.join(root_dir, 'data', 'raw', 'events.csv')
        
        logger.info("Using default file paths for preprocessing")
        return preprocess_data(ratings_path, products_path, events_path)
    else:
        # Process the provided DataFrames
        logger.info("Processing provided DataFrames")
        
        # Generate file paths for saving if needed
        root_dir = Path(__file__).parent.parent
        output_path = os.path.join(root_dir, 'data', 'processed', 'integrated_data.csv')
        
        # Run preprocessing pipeline
        integrated_data = integrate_event_data(ratings_df, events_df)
        
        # Add product information if available
        if products_df is not None:
            integrated_data = pd.merge(
                integrated_data,
                products_df,
                on='product_id',
                how='left'
            )
        
        # Ensure required fields have values
        if 'product_name' not in integrated_data.columns or integrated_data['product_name'].isna().all():
            integrated_data['product_name'] = integrated_data['product_id'].apply(lambda x: f"Product {x}")
        
        if 'product_type' not in integrated_data.columns or integrated_data['product_type'].isna().all():
            integrated_data['product_type'] = 'Generic'
            
        if 'price' not in integrated_data.columns or integrated_data['price'].isna().all():
            integrated_data['price'] = 100  # Default price
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        integrated_data.to_csv(output_path, index=False)
        logger.info(f"Saved integrated data to {output_path}")
        
        return integrated_data

def main():
    # Default paths for data files
    root_dir = Path(__file__).parent.parent
    ratings_path = os.path.join(root_dir, 'data', 'raw', 'dataset.csv')
    products_path = os.path.join(root_dir, 'data', 'raw', 'products.csv')
    events_path = os.path.join(root_dir, 'data', 'raw', 'events.csv')
    
    logger.info("Starting preprocessing pipeline...")
    
    # Create integrated dataset
    integrated_data = preprocess_data(
        ratings_path=ratings_path,
        products_path=products_path,
        events_path=events_path
    )
    
    logger.info(f"Integrated data shape: {integrated_data.shape}")
    logger.info(f"Columns: {integrated_data.columns.tolist()}")
    logger.info("Sample data:")
    logger.info(integrated_data.head())
    
    return integrated_data

if __name__ == "__main__":
    main()