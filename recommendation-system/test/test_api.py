import unittest
from flask import Flask
from main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_recommend_behavior_based(self):
        response = self.app.get('/recommend?user_id=1&case=behavior_based')
        self.assertEqual(response.status_code, 200)
        self.assertIn('recommendations', response.json)

    def test_recommend_hybrid(self):
        response = self.app.get('/recommend?user_id=1&case=hybrid')
        self.assertEqual(response.status_code, 200)
        self.assertIn('recommendations', response.json)

if __name__ == '__main__':
    unittest.main()