import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def generate_recommendations(model, user_id, data, product_details, num_recommendations=3, case='default', product_id=None):
    if case == 'popular':
        # Gợi ý các sản phẩm phổ biến nhất
        popular_products = product_details.sort_values(by='rating', ascending=False).head(num_recommendations)
        return popular_products.to_dict(orient='records')
    
    elif case == 'related' and product_id is not None:
        # Gợi ý các sản phẩm liên quan đến sản phẩm được xem
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

    else:
        # Collaborative Filtering
        try:
            inner_uid = model.trainset.to_inner_uid(user_id)
        except ValueError:
            return []

        all_product_ids = model.trainset.all_items()
        all_product_ids = [model.trainset.to_raw_iid(iid) for iid in all_product_ids]
        rated_product_ids = [j for (j, _) in model.trainset.ur[inner_uid]]
        unrated_product_ids = [pid for pid in all_product_ids if pid not in rated_product_ids]
        predictions = [model.predict(user_id, pid) for pid in unrated_product_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_predictions = predictions[:num_recommendations]
        top_product_ids = [pred.iid for pred in top_predictions]
        
        # Content-Based Filtering
        product_features = product_details[['product_type', 'location', 'price', 'rating']]
        product_features = pd.get_dummies(product_features)
        similarity_matrix = cosine_similarity(product_features)
        product_index = product_details.set_index('product_id')
        
        similar_products = []
        for pid in top_product_ids:
            if pid in product_index.index:
                idx = product_index.index.get_loc(pid)
                similar_indices = similarity_matrix[idx].argsort()[-(num_recommendations+1):-1][::-1]
                similar_products.extend(product_index.iloc[similar_indices].index.tolist())
        
        # Combine results
        recommended_products = product_details[product_details['product_id'].isin(similar_products)]
        
        return recommended_products.to_dict(orient='records')