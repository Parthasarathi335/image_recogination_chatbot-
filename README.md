---
title: Vision Chat
emoji: 🐶
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Vision Chat — Image Classifier

A small Flask web app with a chat-style UI. Upload a photo, and a pretrained
ResNet18 (trained on ImageNet) tells you what it thinks the image shows,
along with a confidence breakdown for its top 3 guesses.

## Project structure

```
project/
├── Dockerfile           # Tells Hugging Face how to build/run the app
├── app.py                # Flask app — the actual website backend
├── chatbot.py             # Standalone CLI prototype, NOT used by the website
├── templates/
│   └── index.html         # Page layout
├── static/
│   ├── script.js            # Upload handling, chat rendering
│   └── style.css            # "Vision scanner" theme
├── requirements.txt
└── dog.jpg                   # Sample test image
```

> **Note:** `chatbot.py` is leftover from an earlier terminal-based prototype.
> It uses ResNet50 and isn't imported by `app.py` or wired into the website
> in any way. It's safe to delete if you don't need a CLI version, or keep
> it as a separate tool — just know it's not part of the live site.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate          
source .venv/bin/activate       

pip install -r requirements.txt
python app.py
```


## How it works

1. The page lets you drag-and-drop or click to choose an image.
2. The image is sent to the `/predict` endpoint as a `multipart/form-data` upload.
3. Flask runs it through ResNet18, computes softmax probabilities, and returns
   the top 3 class guesses with confidence percentages.
4. The frontend renders this as a chat message plus a confidence bar readout.

## Limits

- Max upload size: 8 MB (larger files are rejected with a clear error).
- Runs on CPU by default; will use GPU automatically if CUDA is available.