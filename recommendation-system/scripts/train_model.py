from recommenders.collaborative import train_collaborative_model
from recommenders.content_based import train_content_based_model

def train_all_models(data, product_details):
    collaborative_model, testset = train_collaborative_model(data)
    content_similarity = train_content_based_model(product_details)
    return collaborative_model, content_similarity