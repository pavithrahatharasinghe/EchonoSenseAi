from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Load the random forest model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the sentiment analysis model and vectorizer
sentiment_model = load_model("sentiment_model.h5")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def get_sentiment_class(polarity, threshold=0.1):
    if polarity > threshold:
        return 'positive'
    elif polarity < -threshold:
        return 'negative'
    else:
        return 'neutral'


@app.route('/predict_price', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        feature_values = data['features']
        data_point = np.array(feature_values).reshape(1, -1)
        scaled_data_point = scaler.transform(data_point)
        predicted_close = model.predict(scaled_data_point)
        return jsonify({"predicted_close": predicted_close[0]})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    content = request.json
    text = content['text']
    transformed_text = vectorizer.transform([text]).toarray()
    predicted_polarity = sentiment_model.predict(transformed_text).flatten()[0]
    sentiment_class = get_sentiment_class(predicted_polarity)
    return jsonify({
        "polarity": float(predicted_polarity),
        "sentiment_class": sentiment_class
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)