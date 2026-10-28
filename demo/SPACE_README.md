---
title: Coronary Lesion Classifier
emoji: 🫀
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Detect coronary lesions in X-ray angiography frames
---

# Coronary lesion classifier

Frozen ImageNet MobileNetV2 + MLP head trained on the public CADICA dataset
(Jiménez-Partinen et al., *Expert Systems* 2024, doi:10.1111/exsy.13708) and
evaluated on held-out patients.

Code, evaluation and figures: https://github.com/MarwaSaighi/coronary-lesion-classifier

*Research demonstration only. Not a medical device; not for clinical use.*
