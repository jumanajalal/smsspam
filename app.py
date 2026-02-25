from flask import Flask, request, jsonify
from features import CustomFeatures
import pickle

import re
import unicodedata

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "")

    replacements = {
        "@": "a",
        "3": "e",
        "1": "i",
        "0": "o",
        "$": "s"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    return text

app = Flask(__name__)

# Load model + threshold
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    threshold = data["threshold"]

@app.route("/test")
def test():
    return jsonify({
        "prediction": "Spam",
        "confidence": 88
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "SMS Spam Detection API Running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
        
        message = data["message"]

        # Get probability
        prob_spam = model.predict_proba([message])[0][1]
        prob_ham = 1 - prob_spam
        if any(word in message.lower() for word in ["free", "win", "cash", "click"]) and "http" in message.lower():
            prob_spam += 0.15
        prob_spam = model.predict_proba([message])[0][1]
        prob_ham = 1 - prob_spam
        
        pred = 1 if prob_spam >= threshold else 0

        if pred == 1:
            prediction = "Spam"
            confidence = round(prob_spam * 100, 2)
        else:
            prediction = "Not Spam"
            confidence = round(prob_ham * 100, 2)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "threshold_used": threshold
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # IMPORTANT: allow phone to connect
    app.run()