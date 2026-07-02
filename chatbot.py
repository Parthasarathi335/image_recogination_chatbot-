import torch
from torchvision import models, transforms
from PIL import Image
import requests
import json

class ImageRecognitionChatbot:
    def __init__(self):
        print("Loading model... (this happens once)")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.eval()

        self.labels = models.ResNet50_Weights.DEFAULT.meta["categories"]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("Model loaded! Ready to chat.\n")

    def predict(self, image_path, top_k=3):
        """Run prediction on an image and return top results."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return None, f"Couldn't open image: {e}"

        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_probs, top_idxs = torch.topk(probabilities, top_k)

        results = [
            (self.labels[idx], prob.item() * 100)
            for prob, idx in zip(top_probs, top_idxs)
        ]
        return results, None

    def chat_response(self, image_path):
        results, error = self.predict(image_path)
        if error:
            return f"🤖 {error}"

        response = "🤖 Here's what I think this image shows:\n"
        for label, confidence in results:
            response += f"   • {label}: {confidence:.1f}% confident\n"

        top_label, top_conf = results[0]
        if top_conf > 60:
            response += f"\nI'm fairly confident this is a **{top_label}**."
        else:
            response += f"\nI'm not fully sure, but it might be a **{top_label}**."
        return response


def run_chatbot():
    bot = ImageRecognitionChatbot()
    print("=" * 50)
    print("Image Recognition Chatbot")
    print("Type the path to an image, or 'quit' to exit.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("quit", "exit", "bye"):
            print("🤖 Goodbye!")
            break

        if not user_input:
            continue

        print(bot.chat_response(user_input))


if __name__ == "__main__":
    run_chatbot()