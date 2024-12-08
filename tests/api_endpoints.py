import unittest
from app import app

class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_feed_endpoint_with_username(self):
        response = self.app.get('/feed?username=doey')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('recommendations', data)

    def test_feed_endpoint_with_username_and_category(self):
        response = self.app.get('/feed?username=doey&category_id=1')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('recommendations', data)

    def test_feed_endpoint_missing_username(self):
        response = self.app.get('/feed')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
