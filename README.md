# Video Recommendation System

## Overview

This is a sophisticated video recommendation Flask application that leverages advanced recommendation techniques including:
- Content-based filtering
- Collaborative filtering
- Hybrid recommendation approach

The application fetches data from a social media API and provides personalized video recommendations based on user interactions, content similarity, and collaborative filtering.

## Features

- **Hybrid Recommendation Engine**
  - Content-based recommendations
  - Collaborative filtering
  - Optional category and mood filtering
- **Scalable Data Fetching**
  - Paginated API data retrieval
  - Robust error handling
- **Machine Learning Models**
  - TF-IDF vectorization for content similarity
  - SVD (Singular Value Decomposition) for collaborative filtering
- **Flexible API Endpoint**
  - Personalized recommendations
  - Optional filtering by category and mood

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd video-recommendation-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### API Token
- Replace the `HEADERS` dictionary in the code with your actual Flic API token:
```python
HEADERS = {
    "Flic-Token": "your_actual_token_here"
}
```

## Running the Application

### Development Server
```bash
python app.py
```

The application will start on `http://0.0.0.0:5001`

## API Endpoints

### GET /feed
Retrieve personalized video recommendations

**Query Parameters**:
- `username` (Required): Username to generate recommendations for
- `category_id` (Optional): Filter recommendations by category
- `mood` (Optional): Filter recommendations by mood

**Example Request**:
```
GET /feed?username=johndoe&category_id=123&mood=happy
```

**Example Response**:
```json
{
  "user": "johndoe",
  "category_id": 123,
  "mood": "happy",
  "recommendations": [
    {
      "post_id": "abc123",
      "title": "Interesting Video Title",
      "...": "other video metadata"
    }
    // More recommended videos...
  ]
}
```

## Recommendation Methods

1. **Content-Based Recommendation**
   - Uses TF-IDF vectorization
   - Recommends videos similar to user's previously interacted videos

2. **Collaborative Filtering**
   - Uses SVD algorithm
   - Recommends videos based on ratings and interactions of similar users

3. **Hybrid Recommendation**
   - Combines content-based and collaborative filtering
   - Supports optional category and mood filtering

## Logging

The application uses Python's logging module to log:
- Data fetching
- Model training
- Recommendation generation
- Errors and exceptions

## Performance Metrics

- Includes `precision_at_k` function for evaluating recommendation quality

## Error Handling

- Robust error handling for API requests
- Graceful handling of missing or invalid data
- Comprehensive error responses

## Customization

You can customize:
- Number of recommendations
- Filtering criteria
- Recommendation algorithm weights

## Deployment Considerations

- Use a production WSGI server like Gunicorn
- Set `debug=False` in production
- Secure your API token
- Implement proper authentication

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Specify your license here (e.g., MIT, Apache 2.0)

## Support

For issues or questions, please open a GitHub issue in the repository.