from flask import Flask, request, jsonify
import torch
from model import cnn_model
from utils import preprocess_vector


app = Flask(__name__)

# Load the model
model = cnn_model()
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()


# Define the home route
@app.route("/")
def home():
    return "Welcome to the Rectangle Detection API!"

@app.route("/classify", methods=["POST"])
def classify_vector():
    if "vector" not in request.json:
        return jsonify({"error": "No input vector provided"}), 400

    try:
        # Preprocess the input vector
        input_vector= request.json["vector"]
        input_image = preprocess_vector(input_vector)

        ## Make a prediction
        with torch.no_grad():
            output = model(input_image)
            roi = output.squeeze().tolist()  # Convert output tensor to list (x, y, w, h)

        return jsonify({"roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
