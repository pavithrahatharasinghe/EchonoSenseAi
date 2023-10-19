from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
import numpy as np

from langdetect import detect
import re

app = Flask(__name__)
CORS(app)

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


def is_repetitive(text):
    words = text.split()
    total_words = len(words)
    unique_words = set(words)

    for word in unique_words:
        if words.count(word) / total_words > 0.2:  # if a word appears in more than 5% of the text
            return True
    return False


def validate_text(text):
    # Check if text is empty or not a string
    if not text or not isinstance(text, str):
        return False, "Invalid text input."

    # Minimum Length
    if len(text) < 50:  # for example, less than 50 characters
        return False, "Text is too short."

    # Language Detection
    try:
        if detect(text) != 'en':  # assuming you expect English content
            return False, "Text is not in English."
    except:
        pass  # in case of an error in detection, just pass

    # Check for HTML or Script tags
    if "<script>" in text.lower() or "<html>" in text.lower():
        return False, "Text contains HTML or script tags."



    # Check if a large portion of the text is just numbers
    numeric_chars = re.findall(r'\d', text)
    if len(numeric_chars) / len(text) > 0.5:  # if more than 50% of the content is numeric
        return False, "Text contains too many numbers."

    # Check for repetitive words
    words = text.split()
    total_words = len(words)
    unique_words = set(words)
    for word in unique_words:
        if words.count(word) / total_words > 0.2:  # if a word appears in more than 5% of the text
            return False, "Text seems repetitive."

    # Check for Proper Sentence Structure
    sentences = text.split('.')
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
    if avg_sentence_length > 200:  # for example, if average sentence length is more than 200 characters
        return False, "Text does not have proper sentence structure."

    # Check for All Caps
    all_caps_words = [word for word in words if word.isupper()]
    if len(all_caps_words) / len(words) > 0.2:  # if more than 20% of words are in ALL CAPS
        return False, "Text contains too many words in ALL CAPS."

    return True, ""


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

    # Validate the text
    is_valid, validation_message = validate_text(text)
    if not is_valid:
        return jsonify({"error": validation_message})

    transformed_text = vectorizer.transform([text]).toarray()
    predicted_polarity = sentiment_model.predict(transformed_text).flatten()[0]
    sentiment_class = get_sentiment_class(predicted_polarity)
    return jsonify({
        "polarity": float(predicted_polarity),
        "sentiment_class": sentiment_class
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
