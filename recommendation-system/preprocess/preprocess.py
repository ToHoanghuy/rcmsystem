import pandas as pd

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        if data.empty:
            raise ValueError("No data found in the file.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

def integrate_event_data(ratings, events):
    event_features = events.pivot_table(index=['user_id', 'product_id'], columns='event_type', aggfunc='size', fill_value=0).reset_index()
    dataset = pd.merge(ratings, event_features, on=['user_id', 'product_id'], how='left').fillna(0)
    return dataset