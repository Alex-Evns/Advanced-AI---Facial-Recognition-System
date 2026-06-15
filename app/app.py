from pathlib import Path
import sys
import tempfile

from flask import Flask, render_template, request, jsonify

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = Path(__file__).resolve().parent

# Allow imports from src/
sys.path.append(str(PROJECT_ROOT / "src"))

from inference.predict_image import predict_attributes
from agents.facial_analysis_agent import analyse_face

app = Flask(
    __name__,
    template_folder=str(WEB_APP_DIR / "templates"),
    static_folder=str(WEB_APP_DIR / "static")
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_path = temp_file.name
        image_file.save(image_path)

    predictions = predict_attributes(image_path)
    analysis = analyse_face(predictions)

    return jsonify({
        "predictions": predictions,
        "analysis": analysis
    })


if __name__ == "__main__":
    app.run(debug=True)