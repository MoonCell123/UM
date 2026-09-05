"""Cohort selection and clinical-label loading for fusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

from data_utils.blca_dataset import LABEL_COL as BLCA_LABEL_COL
from data_utils.blca_dataset import load_blca_data
from data_utils.cls_dataset import load_uvm_data


@dataclass(frozen=True)
class CohortSpec:
    name: str
    label_col: str
    patient_id_policy: str


UVM_SPEC = CohortSpec(
    name="UVM",
    label_col="d3m3",
    patient_id_policy="TCGA first three fields; otherwise remove final sample field",
)
BLCA_SPEC = CohortSpec(
    name="BLCA",
    label_col=BLCA_LABEL_COL,
    patient_id_policy="case_id from the BLCA task table",
)


def resolve_cohort_spec(experiment_name: str) -> CohortSpec:
    """Resolve the cohort and fixed task label from a YAML experiment name."""
    normalized = str(experiment_name).strip().upper()
    if normalized in {"UVM", "GME", "OFFLINE_FUSION_BASELINES"}:
        return UVM_SPEC
    if normalized == "BLCA":
        return BLCA_SPEC
    raise ValueError(
        "experiment_name must select a supported cohort: 'UVM' or 'BLCA'. "
        f"Got {experiment_name!r}."
    )


def patient_id_from_slide_id(slide_id: str) -> str:
    parts = str(slide_id).split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3])
    return "-".join(parts[:-1]) if len(parts) > 1 else str(slide_id)


def load_experiment_data(
    experiment_name: str,
    clinical_path: str | Path,
) -> Tuple[pd.DataFrame, list[str], list[int], CohortSpec]:
    """Load the cohort selected by ``experiment_name`` with its fixed label."""
    spec = resolve_cohort_spec(experiment_name)
    if spec.name == "UVM":
        clinical_df, slide_ids, labels = load_uvm_data(
            str(clinical_path), label_col=spec.label_col, cohort_name=spec.name
        )
        clinical_df = clinical_df.copy()
        clinical_df["patient_id"] = clinical_df["slide_id"].map(patient_id_from_slide_id)
    else:
        clinical_df, slide_ids, labels = load_blca_data(
            clinical_path, label_col=spec.label_col, cohort_name=spec.name
        )

    clinical_df["slide_id"] = clinical_df["slide_id"].astype(str)
    clinical_df["patient_id"] = clinical_df["patient_id"].astype(str)
    return clinical_df, slide_ids, labels, spec
