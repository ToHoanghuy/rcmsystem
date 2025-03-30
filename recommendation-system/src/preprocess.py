import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    # Handle missing values
    data = data.ffill()
    return data

def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

def preprocess(filepath):
    data = load_data(filepath)
    cleaned_data = clean_data(data)
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
    event_features = events.pivot_table(index=['user_id', 'product_id'], columns='event_type', aggfunc='size', fill_value=0).reset_index()
    dataset = pd.merge(ratings, event_features, on=['user_id', 'product_id'], how='left').fillna(0)
    return dataset