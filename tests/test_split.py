import pandas as pd
import pytest

from angio_cls.config import EXPECTED_COUNTS, N_PATIENTS, TEST_PATIENTS, VAL_PATIENTS, fold_of
from angio_cls.data import audit_manifest


def test_split_is_disjoint_and_sized():
    assert not set(VAL_PATIENTS) & set(TEST_PATIENTS)
    train = [f"p{i}" for i in range(1, N_PATIENTS + 1) if fold_of(f"p{i}") == "train"]
    assert (len(train), len(VAL_PATIENTS), len(TEST_PATIENTS)) == (30, 6, 6)


def _toy_manifest():
    rows = []
    for i in range(1, N_PATIENTS + 1):
        pid = f"p{i}"
        rows.append(dict(frame_uid=f"{pid}_v1_1", patient_id=pid, video_id="v1", label="lesion", fold=fold_of(pid)))
    return pd.DataFrame(rows)


def test_audit_detects_patient_leakage():
    df = _toy_manifest()
    leak = df.iloc[[0]].assign(frame_uid="leak", fold="test")   # same patient, other split
    with pytest.raises(AssertionError, match="more than one split"):
        audit_manifest(pd.concat([df, leak]))


def test_audit_checks_counts():
    with pytest.raises(AssertionError):
        audit_manifest(_toy_manifest())     # toy counts differ from EXPECTED_COUNTS
    assert sum(sum(v.values()) for v in EXPECTED_COUNTS.values()) == 6126
