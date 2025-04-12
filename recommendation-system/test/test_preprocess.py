import unittest
import pandas as pd
from scripts.preprocess import preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_preprocess_data(self):
        ratings = pd.DataFrame({'user_id': [1, 2], 'product_id': [101, 102], 'rating': [5, 4]})
        products = pd.DataFrame({'product_id': [101, 102], 'product_name': ['A', 'B']})
        events = pd.DataFrame({'user_id': [1], 'event_type': ['view'], 'product_id': [101]})
        
        integrated_data = preprocess_data(ratings, products, events)
        self.assertIn('product_name', integrated_data.columns)
        self.assertEqual(len(integrated_data), 2)

if __name__ == '__main__':
    unittest.main()