import unittest
from app import app


class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_price(self):
        data = {'features': [1.0, 2.0, 3.0, 4.0, 5.0]}  # Replace with actual feature values
        response = self.app.post('/predict_price', json=data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('predicted_close', data)

    def test_predict_sentiment(self):
        data = {'text': 'This is a positive text'}  # Replace with actual text
        response = self.app.post('/predict_sentiment', json=data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('polarity', data)
        self.assertIn('sentiment_class', data)


if __name__ == '__main__':
    unittest.main()
