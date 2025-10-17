from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model and responses
model = joblib.load("model/intent_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

with open("data/responses.json") as f:
    responses = json.load(f)

# Store query logs
logs = []

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat route - receives user message, predicts intent, returns response"""
    data = request.get_json()
    message = data.get("message", "")

    # Predict intent using trained model
    X = vectorizer.transform([message])
    intent = model.predict(X)[0]

    # Get predefined response
    reply = responses.get(intent, "I'm sorry, I didn’t understand that. Could you please rephrase?")

    # Save to logs
    logs.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "intent": intent
    })

    return jsonify({"intent": intent, "response": reply})

@app.route("/analytics", methods=["GET"])
def analytics():
    """Return count of each intent (for dashboard)"""
    from collections import Counter
    intent_count = Counter([log["intent"] for log in logs])
    return jsonify(intent_count)

if __name__ == "__main__":
    print("🚀 Smart Customer Support Assistant running on http://127.0.0.1:5000")
    app.run(debug=True)
