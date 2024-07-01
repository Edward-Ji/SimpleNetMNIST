from PIL import Image

import gradio as gr
import torch
from torchvision.transforms.functional import invert, resize, to_tensor

from simpnet import SimpnetSlim310K

CANVAS_SIZE = (256, 256)

CLASSES = (
    '0 - zero',
    '1 - one',
    '2 - two',
    '3 - three',
    '4 - four',
    '5 - five',
    '6 - six',
    '7 - seven',
    '8 - eight',
    '9 - nine'
)

MODEL_PATH = "models/simpnet_slim_310k.pt"

simpnet_slim = SimpnetSlim310K(1, len(CLASSES))
simpnet_slim.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

def clear():
    return Image.new(mode="L", size=CANVAS_SIZE, color="white")

def predict(sketch):
    image = sketch["composite"]

    simpnet_slim.eval()
    with torch.inference_mode():
        x = invert(resize(to_tensor(image), [32, 32]))
        logits = simpnet_slim(x.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

    return dict(zip(CLASSES, map(torch.Tensor.item, probs.squeeze())))

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(
                value=clear(),
                image_mode="L",
                type="pil",
                transforms=(),
                layers=False,
                canvas_size=CANVAS_SIZE,
                )
        with gr.Column():
            label = gr.Label()
    sketchpad.clear(clear, outputs=sketchpad)
    sketchpad.change(predict, outputs=label, inputs=sketchpad)

if __name__ == "__main__":
    app.launch()
