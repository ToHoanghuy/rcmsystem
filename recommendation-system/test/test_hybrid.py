import unittest
import pandas as pd
import numpy as np
from recommenders.hybrid import hybrid_recommendations

class TestHybrid(unittest.TestCase):
    def test_hybrid_recommendations(self):
        collaborative_model = MockCollaborativeModel()
        content_similarity = np.array([[1, 0.8, 0.6], [0.8, 1, 0.7], [0.6, 0.7, 1]])
        user_id = 1
        products = pd.DataFrame({
            'product_id': [101, 102, 103],
            'product_name': ['A', 'B', 'C']
        })
        recommendations = hybrid_recommendations(collaborative_model, content_similarity, user_id, products)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()

class MockCollaborativeModel:
    def predict(self, uid, iid):
        return type('MockPrediction', (object,), {'est': 4.5})()

collaborative_model = MockCollaborativeModel()