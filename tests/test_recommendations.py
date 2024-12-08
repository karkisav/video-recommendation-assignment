import unittest
from app import recommend_content_based, recommend_collaborative, recommend_hybrid

class TestRecommendations(unittest.TestCase):

    def test_content_based(self):
        user_id = 1  # Mock user ID
        recommendations = recommend_content_based(user_id, num_recommendations=5)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)

    def test_collaborative(self):
        user_id = 1  # Mock user ID
        recommendations = recommend_collaborative(user_id, num_recommendations=5)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)

    def test_hybrid(self):
        user_id = 1  # Mock user ID
        category_id = 10  # Mock category
        mood = "happy"  # Mock mood
        recommendations = recommend_hybrid(user_id, category_id, mood, num_recommendations=5)
        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 5)

if __name__ == '__main__':
    unittest.main()
