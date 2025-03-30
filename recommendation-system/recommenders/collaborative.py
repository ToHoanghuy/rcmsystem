from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_collaborative_model(data):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model, testset