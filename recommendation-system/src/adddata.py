import pandas as pd
import numpy as np

# Tạo dữ liệu cho products.csv
num_products = 1000
product_data = {
    'product_id': range(1, num_products + 1),
    'product_name': [f'Product {i}' for i in range(1, num_products + 1)],
    'product_type': np.random.choice(['Type A', 'Type B', 'Type C'], num_products),
    'description': [f'Description {i}' for i in range(1, num_products + 1)],
    'location': [f'Location {i}' for i in range(1, num_products + 1)],
    'price': np.random.randint(10, 1000, num_products),
    'rating': np.round(np.random.uniform(1, 5, num_products), 1)
}

products_df = pd.DataFrame(product_data)
products_df.to_csv('d:\\python\\recommendation-system\\data\\products.csv', index=False)

# Tạo dữ liệu cho dataset.csv
num_users = 1000
num_events = 10000
user_ids = np.random.randint(1, num_users + 1, num_events)
product_ids = np.random.randint(1, num_products + 1, num_events)
ratings = np.random.randint(1, 6, num_events)

dataset_data = {
    'user_id': user_ids,
    'product_id': product_ids,
    'rating': ratings
}

dataset_df = pd.DataFrame(dataset_data)
dataset_df.to_csv('d:\\python\\recommendation-system\\data\\dataset.csv', index=False)