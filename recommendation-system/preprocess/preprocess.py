import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def analyze_data_distribution(data):
    """
    Phân tích phân phối dữ liệu và trả về báo cáo thống kê
    """
    report = {
        'missing_values': data.isnull().sum().to_dict(),
        'value_counts': {col: data[col].value_counts().to_dict() for col in data.select_dtypes(include=['object']).columns},
        'numeric_stats': data.describe().to_dict()
    }
    print("Data Analysis Report:")
    print(f"Missing values: {report['missing_values']}")
    print(f"Numeric statistics: {report['numeric_stats']}")
    return report

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        if data.empty:
            raise ValueError("No data found in the file.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file.")

def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle missing values based on column type
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
        else:
            data[col] = data[col].fillna(data[col].median() if not data[col].empty else 0)
    
    # Handle outliers using IQR method for numeric columns
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    return data

def normalize_data(data):
    # Use StandardScaler for numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        normalized_numeric = pd.DataFrame(
            scaler.fit_transform(data[numeric_cols]),
            columns=numeric_cols
        )
        
        # Replace columns in original data
        for col in numeric_cols:
            data[col] = normalized_numeric[col]
    
    return data

def preprocess(filepath):
    data = load_data(filepath)
    # Analyze data distribution
    analyze_data_distribution(data)
    # Clean data
    cleaned_data = clean_data(data)
    # Normalize data
    normalized_data = normalize_data(cleaned_data)
    return normalized_data

def load_product_details(filepath):
    try:
        products = pd.read_csv(filepath)
        if products.empty:
            raise ValueError("No data found in the file.")
        return products
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file.")

def load_event_data(filepath):
    try:
        events = pd.read_csv(filepath)
        if events.empty:
            raise ValueError("No data found in the file.")
        return events
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file.")

def integrate_event_data(ratings, events):
    # Check if events dataframe contains user_id and product_id columns
    if 'user_id' not in events.columns or 'product_id' not in events.columns:
        print("Warning: events dataframe must contain user_id and product_id columns")
        return ratings
        
    # Check if events dataframe contains event_type column
    if 'event_type' in events.columns:
        event_features = events.pivot_table(index=['user_id', 'product_id'], columns='event_type', aggfunc='size', fill_value=0).reset_index()
        dataset = pd.merge(ratings, event_features, on=['user_id', 'product_id'], how='left').fillna(0)
    else:
        print("Warning: events dataframe must contain event_type column")
        dataset = ratings
        
    return dataset