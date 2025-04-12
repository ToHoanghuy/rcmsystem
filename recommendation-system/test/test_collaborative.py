import unittest
import pandas as pd
from recommenders.content_based import train_content_based_model

class TestContentBased(unittest.TestCase):
    def test_train_content_based_model(self):
        products = pd.DataFrame({
            'product_id': [1, 2, 3],
            'product_type': ['beach', 'mountain', 'city'],
            'location': ['A', 'B', 'C'],
            'price': [100, 200, 150],
            'rating': [4.5, 4.7, 4.6]
        })
        similarity_matrix = train_content_based_model(products)
        self.assertEqual(similarity_matrix.shape[0], 3)
        self.assertEqual(similarity_matrix.shape[1], 3)

if __name__ == '__main__':
    unittest.main()