"""Gradio demo: coronary lesion detection on a single angiography frame.

Run locally:  PYTHONPATH=src python demo/app.py
"""
from pathlib import Path

import gradio as gr
import torch

from angio_cls.data import eval_transform
from angio_cls.models import LesionClassifier, p_lesion

HERE = Path(__file__).parent
ckpt = torch.load(HERE / "model.pt", map_location="cpu", weights_only=False)
THRESHOLD = ckpt["threshold"]
model = LesionClassifier(ckpt["arch"], pretrain="none")
model.load_state_dict(ckpt["state_dict"])
model.eval_mode()
transform = eval_transform()


def predict(image):
    if image is None:
        return None, ""
    with torch.no_grad():
        p = float(p_lesion(model(transform(image.convert("RGB")).unsqueeze(0)))[0])
    verdict = "**Lesion detected**" if p >= THRESHOLD else "**No lesion detected**"
    note = (f"{verdict}: P(lesion) = {p:.2f}, decision threshold {THRESHOLD:.2f} "
            "(chosen on validation patients to balance sensitivity and specificity).")
    return {"Lesion": p, "No lesion": 1 - p}, note


examples = sorted(str(p) for p in (HERE / "examples").glob("*.png"))

with gr.Blocks(title="Coronary lesion classifier") as demo:
    gr.Markdown(
        "# Coronary lesion classifier\n"
        "Upload an X-ray coronary angiography frame. The model estimates whether it shows a "
        "coronary lesion (narrowing of the artery).\n\n"
        "**Model:** ImageNet-pretrained MobileNetV2 (frozen) with a small MLP head, 2.4M parameters, "
        "trained on the public [CADICA](https://doi.org/10.1111/exsy.13708) dataset and evaluated on "
        "patients never seen during training. Runs on CPU. "
        "[Code and full evaluation on GitHub](https://github.com/MarwaSaighi/coronary-lesion-classifier)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Angiography frame", height=380)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(label="Prediction", num_top_classes=2)
            msg = gr.Markdown()
    gr.Examples(examples, inputs=inp, label="Example frames from held-out test patients "
                "(file name = expert label)")
    gr.Markdown("*Research demonstration only. Not a medical device; not for clinical use.*")
    btn.click(predict, inp, [out, msg])
    inp.upload(predict, inp, [out, msg])

if __name__ == "__main__":
    demo.launch()
