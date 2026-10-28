"""Extract CADICA frames, build the patient-level split and run leakage checks.

    python scripts/prepare_data.py --cadica path/to/CADICA.zip --out data/cadica

Accepts the original CADICA.zip or its extracted folder.
"""
import argparse
import json

from angio_cls.data import audit_manifest, build_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--cadica", required=True, help="CADICA.zip or extracted CADICA folder")
ap.add_argument("--out", default="data/cadica")
a = ap.parse_args()

manifest, boxes = build_manifest(a.cadica, a.out)
report = audit_manifest(manifest)
print(json.dumps(report, indent=2, default=str))
print(f"OK: {len(manifest)} frames, {len(boxes)} lesion boxes -> {a.out}")
