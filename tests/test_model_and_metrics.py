import numpy as np
import pandas as pd
import pytest
import torch

from angio_cls.gradcam import gradcam
from angio_cls.metrics import patient_bootstrap_auc, select_threshold
from angio_cls.models import ARCHS, LesionClassifier


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_shape_and_frozen_backbone(arch):
    m = LesionClassifier(arch, pretrain="none")
    m.eval_mode()
    assert m(torch.zeros(2, 3, 224, 224)).shape == (2, 2)
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert m.n_params()["trainable"] == 128 * m.feature_dim + 386


def test_bn_stats_adapt_but_weights_do_not():
    m = LesionClassifier("mobilenet_v2", pretrain="none")
    bn = next(x for x in m.backbone.modules() if isinstance(x, torch.nn.BatchNorm2d))
    w0, rm0 = bn.weight.clone(), bn.running_mean.clone()
    m.train_mode()
    m(torch.randn(4, 3, 224, 224))
    assert torch.equal(bn.weight, w0) and not torch.equal(bn.running_mean, rm0)


def test_gradcam_returns_normalised_map():
    m = LesionClassifier("resnet18", pretrain="none")
    cam, p = gradcam(m, torch.randn(1, 3, 224, 224))
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1 and 0 <= p <= 1


def test_threshold_selected_on_validation():
    y = np.array([1, 1, 1, 0, 0, 0]); p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert 0.3 < select_threshold(y, p) <= 0.7


def test_bootstrap_resamples_patients():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_id": np.repeat([f"p{i}" for i in range(6)], 20),
                       "true_lesion": rng.integers(0, 2, 120), "prob_lesion": rng.random(120)})
    lo, hi = patient_bootstrap_auc(df, n_reps=200)
    assert 0 <= lo < hi <= 1
