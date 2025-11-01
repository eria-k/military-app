from flask import Flask, request, jsonify, render_template
import os, numpy as np

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("sensor_data", [])
    return jsonify({
        "success": True,
        "message": "Model not integrated yet — scaffold working",
        "received_features": len(data)
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
