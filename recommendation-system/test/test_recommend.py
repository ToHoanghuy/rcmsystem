import unittest
import pandas as pd
from recommenders.collaborative import train_collaborative_model
from recommenders.content_based import train_content_based_model

class TestRecommend(unittest.TestCase):
    def test_collaborative_model(self):
        data = pd.DataFrame({'user_id': [1, 2], 'product_id': [101, 102], 'rating': [5, 4]})
        model, testset = train_collaborative_model(data)
        self.assertIsNotNone(model)

    def test_content_based_model(self):
        products = pd.DataFrame({
            'product_id': [1, 2, 3],
            'product_type': ['beach', 'mountain', 'city'],
            'location': ['A', 'B', 'C'],
            'price': [100, 200, 150],
            'rating': [4.5, 4.7, 4.6]
        })
        similarity_matrix = train_content_based_model(products)
        self.assertEqual(similarity_matrix.shape[0], 3)

if __name__ == '__main__':
    unittest.main()