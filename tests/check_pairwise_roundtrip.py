"""Optional cross-environment integration: Torch producer -> NumPy analyzer.

Usage: torch-python tests/check_pairwise_roundtrip.py --analysis-python <python>
The analysis interpreter must have numpy and pandas. All data are synthetic.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

from test_pairwise_cache import TinyReference, serialize
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-python", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = TinyReference()
    features = {name: torch.tensor([[float(index + 1)]]) for index, name in enumerate(model.encoder_names)}
    cache = {}
    _, _, interactions = model.intervention_attribution(features, {}, replacement_strategy="zero",
                                                       compute_interactions=True, interaction_target_class=1,
                                                       coalition_logits_out=cache)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        source = directory / "pairwise_state_logits.csv"
        rows = []
        for i, left in enumerate(model.encoder_names):
            for j in range(i + 1, len(model.encoder_names)):
                right = model.encoder_names[j]
                rows.append(dict(schema_version=1, slide_id="SYNTHETIC_001", label=0, fold=1, split="val",
                                 encoder_i=left, encoder_j=right, target_class=1,
                                 interaction=float(interactions[i, j]), checkpoint="synthetic.pt",
                                 baseline_file="unused_zero.pt", replacement_strategy="zero",
                                 gaussian_std_scale=1., reference_forward="mean_fusion",
                                 **serialize(cache, left, right)))
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output = directory / "analysis"
        completed = subprocess.run([args.analysis_python, str(root / "code/evaluation/analyze_pairwise_states.py"),
                                    "--input-files", str(source), "--output-dir", str(output), "--no-plots"],
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["n_observations"] == 3
        with (output / "pairwise_state_diagnostics.csv").open(encoding="utf-8-sig", newline="") as handle:
            diagnostics = list(csv.DictReader(handle))
        for row in diagnostics:
            assert abs(float(row["interaction_analysis"]) + float(row["interaction"])) < 1e-5
        print("PASS: 3 production-cache pairs -> exact CSV -> standalone analyzer; class-0 sign and reconstruction verified.")


if __name__ == "__main__":
    main()
