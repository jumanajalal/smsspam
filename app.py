from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pipeline model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    
    if request.method == "POST":
        message = request.form["message"]
        
        # Direct prediction (pipeline handles TF-IDF internally)
        pred = model.predict([message])[0]
        prob = model.predict_proba([message])[0][1]
        
        prediction = "Spam 🚨" if pred == 1 else "Not Spam ✅"
        confidence = round(prob * 100, 2)
    
    return render_template("index.html", prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)