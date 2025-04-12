import unittest
import pandas as pd
from recommenders.event_based import recommend_based_on_events

class TestEventBased(unittest.TestCase):
    def test_recommend_based_on_events(self):
        events = pd.DataFrame({
            'user_id': [1, 1, 2],
            'event_type': ['view', 'purchase', 'view'],
            'product_id': [101, 102, 103]
        })
        products = pd.DataFrame({
            'product_id': [101, 102, 103],
            'product_name': ['A', 'B', 'C'],
            'product_type': ['type1', 'type2', 'type1'],
            'price': [100, 200, 150],
            'rating': [4.5, 4.7, 4.6]
        })
        recommendations = recommend_based_on_events(1, events, products, top_n=2)
        self.assertEqual(len(recommendations), 2)

if __name__ == '__main__':
    unittest.main()