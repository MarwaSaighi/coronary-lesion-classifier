"""Fixed constants for the CADICA lesion-classification study.

These values were frozen before any model was trained and are shared by data
preparation, training, evaluation and the demo.
"""

# Class index order used by every model in this repo (output 0 = lesion).
CLASS_TO_IDX = {"lesion": 0, "non_lesion": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Patient-level split of the 42 CADICA patients: drawn once with a fixed seed
# and frozen. Every frame and video of a patient lives in exactly one split.
VAL_PATIENTS = ["p3", "p5", "p11", "p24", "p25", "p42"]
TEST_PATIENTS = ["p4", "p12", "p23", "p26", "p36", "p40"]
N_PATIENTS = 42

# Expected frame counts after preparation (used as a data-integrity check).
EXPECTED_COUNTS = {
    "train": {"lesion": 2589, "non_lesion": 1521},
    "val": {"lesion": 788, "non_lesion": 263},
    "test": {"lesion": 619, "non_lesion": 346},
}

# Grayscale intensity statistics computed on the TRAINING split only.
NORM_MEAN = [0.5121431350708008] * 3
NORM_STD = [0.13775277137756348] * 3
INPUT_SIZE = 224

# torchvision ImageNet weights used for each backbone.
IMAGENET_WEIGHTS = {
    "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
    "mobilenet_v2": "MobileNet_V2_Weights.IMAGENET1K_V2",
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    "shufflenet_v2_x0_5": "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1",
    "squeezenet1_1": "SqueezeNet1_1_Weights.IMAGENET1K_V1",
}


def fold_of(patient_id: str) -> str:
    if patient_id in VAL_PATIENTS:
        return "val"
    if patient_id in TEST_PATIENTS:
        return "test"
    return "train"
