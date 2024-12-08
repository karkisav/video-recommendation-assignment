from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import logging
from surprise import accuracy

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)

BASE_URL = "https://api.socialverseapp.com"
HEADERS = {
    "Flic-Token": "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"
}

# --- Data Fetching and Preprocessing ---
def fetch_data(endpoint, max_pages=10, extra_params=None):
    """
    Fetch paginated data from a given API endpoint.
    """
    data = []
    page = 1

    if extra_params is None:
        extra_params = {}

    while page <= max_pages:
        # Add pagination parameters
        params = {"page": page, "page_size": 1000}
        params.update(extra_params)

        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Extract the 'posts' key from the response JSON
            page_data = response.json().get('posts', [])
            if not page_data:  # Stop if no 'posts' data is returned
                page_data = response.json().get('users', [])
            if not page_data:   
                break
            
            data.extend(page_data)
            page += 1
        except requests.RequestException as e:
            logger.error(f"Error fetching data from {endpoint}: {e}")
            break

    return pd.DataFrame(data)

def load_data():
    """
    Load and preprocess data from various endpoints
    """
    global all_users, all_posts, rated_posts
    extra_params = {
        "resonance_algorithm": "resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
    }
    
    try:
        viewed_posts = fetch_data(endpoint="posts/view", extra_params=extra_params)
        liked_posts = fetch_data(endpoint="posts/like", extra_params=extra_params)
        inspired_posts = fetch_data(endpoint="posts/inspire", extra_params=extra_params)
        rated_posts = fetch_data(endpoint="posts/rating", extra_params=extra_params)
        all_posts = fetch_data(endpoint="posts/summary/get")
        all_users = fetch_data(endpoint="users/get_all")

        # Rename id columns
        all_posts.rename(columns={'id': 'post_id'}, inplace=True)
        all_users.rename(columns={'id': 'user_id'}, inplace=True)

        # Merge interactions
        interactions = pd.concat([viewed_posts, liked_posts, inspired_posts, rated_posts])
        
        # Merge with metadata
        posts_with_metadata = pd.merge(interactions, all_posts, on='post_id', how='inner')
        user_data = pd.merge(posts_with_metadata, all_users, on='user_id', how='inner')

        logger.info(f"Total users: {len(all_users)}")
        logger.info(f"Total posts: {len(all_posts)}")
        logger.info(f"Total interactions: {len(interactions)}")
        logger.info(f"All posts columns: {all_posts.columns.tolist()}")

        return user_data, all_posts, rated_posts

    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

# Preprocessing and Model Training
def preprocess_data(user_data, all_posts):
    """
    Preprocess data for recommendation system
    """
    # Numeric feature scaling
    scaler = MinMaxScaler()
    numeric_features = ['upvote_count', 'view_count', 'rating_percent', 'average_rating']
    user_data[numeric_features] = scaler.fit_transform(user_data[numeric_features])

    # Text preprocessing
    all_posts['title'] = all_posts['title'].fillna('').astype(str)
    all_posts['post_summary'] = all_posts['post_summary'].fillna('').astype(str)

    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    content_combined = all_posts['title'] + ' ' + all_posts['post_summary']
    all_posts['content_vector'] = list(tfidf.fit_transform(content_combined).toarray())
    
    # Similarity matrix
    similarity_matrix = cosine_similarity(np.array(all_posts['content_vector'].tolist()))

    return user_data, all_posts, similarity_matrix

def train_recommendation_model(rated_posts):
    """
    Train collaborative filtering model
    """
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(rated_posts[['user_id', 'post_id', 'rating_percent']], reader)
    
    # Split data for training and testing
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train SVD model
    model = SVD()
    model.fit(trainset)
    
    return model, trainset, testset

# Load and preprocess data
try:
    user_data, all_posts, rated_posts = load_data()
    user_data, all_posts, similarity_matrix = preprocess_data(user_data, all_posts)
    model, trainset, testset = train_recommendation_model(rated_posts)
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

# --- Recommendation Functions ---
def recommend_content_based(user_id, num_recommendations=10):
    """
    Content-based recommendation
    """
    user_interactions = user_data[user_data['user_id'] == user_id]
    user_posts = user_interactions['post_id'].unique()

    similar_posts = []
    for post_id in user_posts:
        post_index = all_posts[all_posts['post_id'] == post_id].index[0]
        similar_posts += list(enumerate(similarity_matrix[post_index]))

    similar_posts = sorted(similar_posts, key=lambda x: x[1], reverse=True)
    recommended_post_ids = [
        all_posts.iloc[i[0]]['post_id']
        for i in similar_posts
        if all_posts.iloc[i[0]]['post_id'] not in user_posts
    ]
    return recommended_post_ids[:num_recommendations]

def recommend_collaborative(user_id, num_recommendations=10):
    """
    Collaborative filtering recommendation
    """
    user_rated_posts = rated_posts[rated_posts['user_id'] == user_id]['post_id'].unique()
    all_post_ids = all_posts['post_id'].unique()
    unrated_posts = [post_id for post_id in all_post_ids if post_id not in user_rated_posts]

    predictions = [model.predict(user_id, post_id) for post_id in unrated_posts]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    recommended_post_ids = [pred.iid for pred in predictions[:num_recommendations]]
    return recommended_post_ids

# --- Recommendation Functions ---
def recommend_content_based(user_id, num_recommendations=10):
    """
    Content-based recommendation
    """
    user_interactions = user_data[user_data['user_id'] == user_id]
    user_posts = user_interactions['post_id'].unique()

    similar_posts = []
    for post_id in user_posts:
        post_index = all_posts[all_posts['post_id'] == post_id].index[0]
        similar_posts += list(enumerate(similarity_matrix[post_index]))

    similar_posts = sorted(similar_posts, key=lambda x: x[1], reverse=True)
    recommended_post_ids = [
        all_posts.iloc[i[0]]['post_id']
        for i in similar_posts
        if all_posts.iloc[i[0]]['post_id'] not in user_posts
    ]
    return recommended_post_ids[:num_recommendations]

def recommend_collaborative(user_id, num_recommendations=10):
    """
    Collaborative filtering recommendation
    """
    user_rated_posts = rated_posts[rated_posts['user_id'] == user_id]['post_id'].unique()
    all_post_ids = all_posts['post_id'].unique()
    unrated_posts = [post_id for post_id in all_post_ids if post_id not in user_rated_posts]

    predictions = [model.predict(user_id, post_id) for post_id in unrated_posts]
    predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

    recommended_post_ids = [pred.iid for pred in predictions[:num_recommendations]]
    return recommended_post_ids

def recommend_hybrid(user_id, category_id=None, mood=None, num_recommendations=10):
    """
    Hybrid recommendation with optional filtering
    """
    filtered_posts = all_posts.copy()
    
    # Apply optional filters
    if category_id:
        try:
            # Assuming category is a nested dictionary
            filtered_posts = filtered_posts[filtered_posts['category'].apply(lambda x: x.get('id') == int(category_id))]
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not filter by category. Error: {e}")
    if mood:
        if 'mood' in filtered_posts.columns:
            filtered_posts = filtered_posts[filtered_posts['mood'] == mood]
        else:
            logger.warning("'mood' column not found in posts data. Skipping mood filter.")

    # Generate recommendations
    content_recs = recommend_content_based(user_id, num_recommendations)
    collab_recs = recommend_collaborative(user_id, num_recommendations)

    # Combine and deduplicate recommendations
    recs = list(set(content_recs) | set(collab_recs))
    return recs[:num_recommendations]

def precision_at_k(actual, predicted, k):
    actual = set(actual[:k])
    predicted = set(predicted[:k])
    return len(actual & predicted) / min(len(predicted), k)


def get_user_id(username, all_users):
    print(username)
    user = all_users[all_users['username'] == username]
    if not user.empty:
        return user.iloc[0]['user_id']
    else:
        return None

@app.route('/feed', methods=['GET'])
def get_recommendations():
    username = request.args.get('username')
    category_id = request.args.get('category_id')
    mood = request.args.get('mood')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    try:
        user_id = get_user_id(username, all_users)
        if not user_id:
            return jsonify({"error": f"User '{username}' not found"}), 404

        # Convert category_id to integer if provided
        category_id = int(category_id) if category_id else None

        recommendations = recommend_hybrid(
            user_id=user_id,
            category_id=category_id,
            mood=mood,
            num_recommendations=10
        )

        # Exclude non-serializable fields like 'content_vector' from all_posts
        recommended_posts = all_posts[all_posts['post_id'].isin(recommendations)].copy()
        recommended_posts.drop(columns=['content_vector'], errors='ignore', inplace=True)

        response = {
            "user": username,
            "category_id": category_id,
            "mood": mood,
            "recommendations": recommended_posts.to_dict(orient='records')
        }
        return jsonify(response), 200

    except ValueError:
        return jsonify({"error": "Invalid category_id. Must be an integer."}), 400
    except Exception as e:
        logger.error(f"Error in /feed: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during recommendation generation"}), 500


@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Video Recommendation API!",
        "endpoints": {
            "GET /feed": "Get personalized recommendations. Query parameters: username, category_id, mood."
        }
    }), 200

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)