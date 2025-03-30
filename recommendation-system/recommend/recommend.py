import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def generate_recommendations(model, user_id, data, product_details, num_recommendations=10, case='default', product_id=None):
    if case == 'popular':
        popular_products = product_details.sort_values(by='rating', ascending=False).head(num_recommendations)
        return popular_products.to_dict(orient='records')
    
    elif case == 'related' and product_id is not None:
        product_features = product_details[['product_type', 'location', 'price', 'rating']]
        product_features = pd.get_dummies(product_features)
        similarity_matrix = cosine_similarity(product_features)
        product_index = product_details.set_index('product_id')
        
        if product_id in product_index.index:
            idx = product_index.index.get_loc(product_id)
            similar_indices = similarity_matrix[idx].argsort()[-(num_recommendations+1):-1][::-1]
            related_products = product_index.iloc[similar_indices].reset_index()
            return related_products.to_dict(orient='records')
        else:
            return []

    elif case == 'user_history' and user_id is not None:
        user_history = data[data['user_id'] == user_id]
        if user_history.empty:
            return []
        top_products = user_history.sort_values(by='rating', ascending=False).head(num_recommendations)
        return product_details[product_details['product_id'].isin(top_products['product_id'])].to_dict(orient='records')

    else:
        return []