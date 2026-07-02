"""
Image Recognition Chatbot - Flask backend
Loads a pretrained ResNet18 (ImageNet weights) and classifies uploaded images.
Swap the model-loading section with your own trained model if you have one.
"""

import io
import random

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
model.to(device)

CLASSES = weights.meta["categories"]

preprocess = weights.transforms()

HIGH_CONF_REPLIES = [
    "That's a {label}. I'm {conf}% sure about this one.",
    "Looks like a {label} to me — {conf}% confidence.",
    "I'd classify this as a {label} ({conf}% confidence).",
]
LOW_CONF_REPLIES = [
    "I think this might be a {label}, but I'm only {conf}% confident.",
    "Hard to tell, but my best guess is {label} ({conf}% confidence).",
    "I'm not very sure, but it could be a {label} — {conf}% confidence.",
]


def classify_image(image: Image.Image, top_k: int = 3):
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_idxs = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        results.append(
            {
                "label": CLASSES[idx].replace("_", " "),
                "confidence": round(prob.item() * 100, 2),
            }
        )
    return results


def build_chat_message(top_result: dict) -> str:
    template_pool = HIGH_CONF_REPLIES if top_result["confidence"] >= 50 else LOW_CONF_REPLIES
    template = random.choice(template_pool)
    return template.format(label=top_result["label"], conf=top_result["confidence"])

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    results = classify_image(image, top_k=3)
    message = build_chat_message(results[0])

    return jsonify({"message": message, "predictions": results})


@app.errorhandler(413)
def file_too_large(_e):
    return jsonify({"error": "Image is too large (max 8 MB)."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)