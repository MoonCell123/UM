"""Clinical-label loading for the TCGA-BLCA TP53 mutation task."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


LABEL_COL = "tp53_mutation"


def _read_table(clinical_path: str | Path) -> pd.DataFrame:
    path = Path(clinical_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.read_excel(path)


def _patient_id_from_tcga_slide(slide_id: str) -> str:
    parts = str(slide_id).split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3])
    return "-".join(parts[:-1]) if len(parts) > 1 else str(slide_id)


def load_blca_data(
    clinical_path: str | Path,
    label_col: str = LABEL_COL,
    cohort_name: str = "BLCA",
) -> Tuple[pd.DataFrame, list[str], list[int]]:
    """Load BLCA slide labels from the TP53 task table.

    The canonical task table stores a WSI filename, a patient/case identifier,
    and a binary ``label`` (0: TP53 wildtype; 1: TP53 mutant). Feature files
    are named with the filename stem, so ``slide_id`` is derived from
    ``filename`` when no explicit slide identifier is provided.
    """
    if label_col != LABEL_COL:
        raise ValueError(f"BLCA uses the fixed label column {LABEL_COL!r}, got {label_col!r}.")

    table = _read_table(clinical_path).copy()
    if "slide_id" not in table.columns:
        if "filename" not in table.columns:
            raise ValueError(
                f"BLCA task table must contain 'filename' or 'slide_id'. Columns: {table.columns.tolist()}"
            )
        table["slide_id"] = table["filename"].map(lambda value: Path(str(value)).stem)
    table["slide_id"] = table["slide_id"].astype(str)

    source_label = LABEL_COL if LABEL_COL in table.columns else "label"
    if source_label not in table.columns:
        raise ValueError(
            f"BLCA task table must contain 'label' or {LABEL_COL!r}. Columns: {table.columns.tolist()}"
        )
    table[LABEL_COL] = pd.to_numeric(table[source_label], errors="coerce")
    valid = table[table[LABEL_COL].isin([0, 1])].copy()
    valid[LABEL_COL] = valid[LABEL_COL].astype(int)

    if "case_id" in valid.columns:
        valid["patient_id"] = valid["case_id"].astype(str)
    else:
        valid["patient_id"] = valid["slide_id"].map(_patient_id_from_tcga_slide)

    duplicate_labels = valid.groupby("slide_id")[LABEL_COL].nunique()
    conflicting = duplicate_labels[duplicate_labels > 1]
    if not conflicting.empty:
        raise ValueError(
            "BLCA task table assigns conflicting labels to slides, e.g. "
            f"{conflicting.index[:5].tolist()}"
        )
    valid = valid.drop_duplicates("slide_id", keep="first")
    if valid.empty:
        raise ValueError(f"No valid binary TP53 labels found in {clinical_path}.")

    slide_ids = valid["slide_id"].tolist()
    labels = valid[LABEL_COL].tolist()
    print(f"[{cohort_name}] clinical rows: {len(table)}, valid slides: {len(valid)}")
    print(f"  TP53 wildtype (label=0): {labels.count(0)} | TP53 mutant (label=1): {labels.count(1)}")
    return valid, slide_ids, labels
