import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

def classify(img):
    if img is None:
        return {"healthy": 0.0, "diseased": 0.0}
    r = model.predict(img, verbose=False)[0]
    pred = r.names[r.probs.top1]
    conf = float(r.probs.top1conf)
    return {pred: conf}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a fish photo"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🐟 Fish Disease Classifier",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)