"""Static Beacon construction and similarity scoring for middle-fusion models.

Stage policy:
    1. Train ProjectionHead with the downstream model.
    2. Freeze ProjectionHead.
    3. Use train-split embeddings only to project -> build Beacon -> save .pt.

The saved Beacon is then used as a static Global Prior for validation/test and
later model training stages.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = PROJECT_ROOT / "code" / "architecture"
if str(ARCHITECTURE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHITECTURE_DIR))

from projection_head import MultiEncoderProjectionHead


FEATURE_KEYS = ("feats", "features")


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the last tensor dimension."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class BeaconAccumulator:
    """Streaming accumulator for train-set Beacon vectors.
    """

    def __init__(self, target_dim: int = 512, eps: float = 1e-8, device: str | torch.device = "cpu"):
        self.target_dim = int(target_dim)
        self.eps = float(eps)
        self.device = torch.device(device)
        self._sums: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    @torch.no_grad()
    def update(self, encoder_name: str, projected_embeddings: torch.Tensor) -> None:
        """Accumulate projected embeddings for one encoder.

        Args:
            encoder_name: Name matching the feature_dir/model encoder.
            projected_embeddings: Tensor with shape [..., target_dim].
        """
        if projected_embeddings.shape[-1] != self.target_dim:
            raise ValueError(
                f"{encoder_name}: expected last dim {self.target_dim}, "
                f"got {projected_embeddings.shape[-1]}"
            )

        h = projected_embeddings.detach().to(self.device).reshape(-1, self.target_dim)
        h = l2_normalize(h, eps=self.eps)

        if encoder_name not in self._sums:
            self._sums[encoder_name] = torch.zeros(self.target_dim, device=self.device)
            self._counts[encoder_name] = 0

        self._sums[encoder_name] += h.sum(dim=0)
        self._counts[encoder_name] += int(h.shape[0])

    def encoder_means(self) -> Dict[str, torch.Tensor]:
        """Return per-encoder mean normalized embeddings, i.e. bar{h}_m."""
        means = {}
        for encoder_name, total in self._sums.items():
            count = self._counts[encoder_name]
            if count <= 0:
                raise ValueError(f"{encoder_name}: cannot compute mean with count={count}")
            means[encoder_name] = total / count
        return means

    def compute(self, normalize_beacon: bool = True) -> torch.Tensor:
        """Compute global Beacon B by averaging per-encoder means."""
        means = self.encoder_means()
        if not means:
            raise RuntimeError("BeaconAccumulator has no embeddings. Call update() first.")
        #beacon = torch.stack(list(means.values()), dim=0).mean(dim=0)
        means = [
            l2_normalize(means[k])
            for k in sorted(means.keys())
        ]
        beacon = torch.stack(means).mean(dim=0)
        if normalize_beacon:
            beacon = l2_normalize(beacon, eps=self.eps)
        return beacon

    def summary(self) -> pd.DataFrame:
        rows = []
        for encoder_name in sorted(self._counts):
            rows.append({
                "encoder_name": encoder_name,
                "count": self._counts[encoder_name],
                "sum_norm": float(self._sums[encoder_name].norm().detach().cpu()),
            })
        return pd.DataFrame(rows)


def beacon_similarity(
    projected_embeddings: torch.Tensor,
    beacon: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
    use_cosine: bool = False,
) -> torch.Tensor:
    """Compute similarity between projected embeddings and Beacon.

    Dot-product mode is used
    Cosine mode is available as the noted backup choice.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if projected_embeddings.shape[-1] != beacon.shape[-1]:
        raise ValueError(
            f"Dim mismatch: embeddings dim={projected_embeddings.shape[-1]}, "
            f"beacon dim={beacon.shape[-1]}"
        )

    b = beacon.to(projected_embeddings.device)
    if use_cosine:
        h = l2_normalize(projected_embeddings, eps=eps)
        b = l2_normalize(b, eps=eps)
        return torch.sum(h * b, dim=-1) / temperature

    d = projected_embeddings.shape[-1]
    return torch.sum(projected_embeddings * b, dim=-1) / (math.sqrt(d) * temperature)


def get_dataset_name(h5_path: Path) -> str:
    with h5py.File(h5_path, "r") as f:
        for key in FEATURE_KEYS:
            if key in f:
                return key
        candidates = [key for key in f.keys() if key != "coords"]
        if not candidates:
            raise KeyError(f"{h5_path}: no feature dataset found. Keys: {list(f.keys())}")
        return candidates[0]


def read_h5_features(h5_path: Path, dataset_key: str | None = None) -> np.ndarray:
    key = dataset_key or get_dataset_name(h5_path)
    with h5py.File(h5_path, "r") as f:
        features = np.asarray(f[key][:], dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"{h5_path}: expected feature shape [N, D], got {features.shape}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def infer_input_dims(manifest: pd.DataFrame, fold: int | None = None) -> Dict[str, int]:
    """Infer input dimensions for each encoder from manifest feature_dim."""
    rows = manifest if fold is None else manifest[manifest["fold"] == fold]
    if rows.empty:
        raise ValueError(f"No manifest rows found for fold={fold}")

    input_dims: Dict[str, int] = {}
    for feature_dir, group in rows.groupby("feature_dir"):
        dims = sorted(group["feature_dim"].astype(int).unique().tolist())
        if len(dims) != 1:
            raise ValueError(f"{feature_dir}: expected one feature_dim, got {dims}")
        input_dims[str(feature_dir)] = int(dims[0])
    return input_dims


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict-like object, got {type(checkpoint)}")

    for key in (
        "projection_heads",
        "projection_head",
        "projector",
        "model_state_dict",
        "state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError(
        "Could not find a projection state_dict. Expected one of: "
        "projection_heads, projection_head, projector, model_state_dict, state_dict."
    )


def _state_dict_candidates(state_dict: Mapping[str, torch.Tensor]) -> Iterable[Dict[str, torch.Tensor]]:
    original = dict(state_dict)
    yield original

    prefixes = (
        "module.",
        "projection_heads.",
        "projection_head.",
        "projector.",
        "model.projection_heads.",
        "model.projection_head.",
        "model.projector.",
    )
    for prefix in prefixes:
        stripped = {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in original.items()
        }
        yield stripped

    if any("heads." in key for key in original):
        yield {
            key[key.index("heads."):] if "heads." in key else key: value
            for key, value in original.items()
        }


def load_frozen_projection_heads(
    checkpoint_path: str | Path,
    input_dims: Mapping[str, int],
    target_dim: int = 512,
    dropout: float = 0.0,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> MultiEncoderProjectionHead:
    """Load Stage-1 ProjectionHead weights and freeze them for Beacon building."""
    device = torch.device(device)
    projection_heads = MultiEncoderProjectionHead(
        input_dims=input_dims,
        target_dim=target_dim,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)

    last_error: RuntimeError | None = None
    incompatible = None
    model_keys = set(projection_heads.state_dict().keys())
    for candidate in _state_dict_candidates(state_dict):
        matched_keys = model_keys.intersection(candidate.keys())
        if not matched_keys:
            continue
        try:
            incompatible = projection_heads.load_state_dict(candidate, strict=strict)
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    if incompatible is None:
        raise KeyError(
            "No checkpoint keys matched MultiEncoderProjectionHead. Expected keys like "
            "'heads.<feature_dir>.proj.0.weight'."
        )

    projection_heads.eval()
    for param in projection_heads.parameters():
        param.requires_grad_(False)

    if incompatible is not None and not strict:
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f"[Warning] Missing projection keys: {missing[:10]}")
        if unexpected:
            print(f"[Warning] Unexpected checkpoint keys: {unexpected[:10]}")
    return projection_heads


def _project_in_batches(
    raw_features: np.ndarray,
    projection_head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> Iterable[torch.Tensor]:
    projection_head.eval()
    with torch.no_grad():
        for start in range(0, raw_features.shape[0], batch_size):
            batch = torch.from_numpy(raw_features[start:start + batch_size]).float().to(device)
            yield projection_head(batch)


@torch.no_grad()
def build_beacon_from_manifest(
    manifest_path: str | Path,
    projection_heads: Mapping[str, torch.nn.Module],
    fold: int,
    target_dim: int = 512,
    split: str = "train",
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
    normalize_beacon: bool = True,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """Build a fold-specific Beacon from manifest rows.
    """
    device = torch.device(device)
    manifest = pd.read_csv(manifest_path)
    rows = manifest[(manifest["fold"] == fold) & (manifest["split"] == split)].copy()
    if rows.empty:
        raise ValueError(f"No rows found for fold={fold}, split={split} in {manifest_path}")

    missing_heads = sorted(set(rows["feature_dir"]) - set(projection_heads))
    if missing_heads:
        raise KeyError(f"Missing projection heads for feature_dir values: {missing_heads}")

    accumulator = BeaconAccumulator(target_dim=target_dim, device=device)
    for _, row in rows.iterrows():
        feature_dir = str(row["feature_dir"])
        h5_path = Path(row["h5_path"])
        dataset_key = str(row["dataset_key"]) if "dataset_key" in row and pd.notna(row["dataset_key"]) else None
        raw_features = read_h5_features(h5_path, dataset_key=dataset_key)
        head = projection_heads[feature_dir].to(device)
        for projected in _project_in_batches(raw_features, head, device, batch_size):
            accumulator.update(feature_dir, projected)

    beacon = accumulator.compute(normalize_beacon=normalize_beacon)
    return beacon, accumulator.summary()


@torch.no_grad()
def build_static_beacon_from_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    fold: int,
    target_dim: int = 512,
    split: str = "train",
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
    dropout: float = 0.0,
    normalize_beacon: bool = True,
    strict_load: bool = False,
) -> Tuple[torch.Tensor, pd.DataFrame, Dict[str, int]]:
    """Stage-2 helper: load frozen ProjectionHeads and build static Beacon."""
    manifest = pd.read_csv(manifest_path)
    input_dims = infer_input_dims(manifest, fold=fold)
    projection_heads = load_frozen_projection_heads(
        checkpoint_path=checkpoint_path,
        input_dims=input_dims,
        target_dim=target_dim,
        dropout=dropout,
        device=device,
        strict=strict_load,
    )
    beacon, summary = build_beacon_from_manifest(
        manifest_path=manifest_path,
        projection_heads=projection_heads.heads,
        fold=fold,
        target_dim=target_dim,
        split=split,
        batch_size=batch_size,
        device=device,
        normalize_beacon=normalize_beacon,
    )
    return beacon, summary, input_dims


def save_beacon(
    beacon: torch.Tensor,
    summary: pd.DataFrame,
    output_dir: str | Path,
    fold: int,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Save Beacon tensor and construction summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"fold": int(fold), "beacon": beacon.detach().cpu()}
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    torch.save(payload, output_dir / f"fold_{fold}_beacon.pt")
    summary.to_csv(output_dir / f"fold_{fold}_beacon_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 static Beacon builder. Load a Stage-1 trained ProjectionHead "
            "checkpoint, freeze projection, project train split, and save Beacon .pt."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("output/Middle_Fusion_Manifests/middle_fusion_manifest.csv"))
    parser.add_argument("--projection-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/Beacon"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--normalize-beacon", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    manifest = pd.read_csv(args.manifest)
    rows = manifest[(manifest["fold"] == args.fold) & (manifest["split"] == args.split)]
    print("=" * 80)
    print("Stage-2 static Beacon builder")
    print("=" * 80)
    print(f"Manifest: {args.manifest}")
    print(f"fold={args.fold}, split={args.split}")
    print(f"Rows: {len(rows)}")
    print(f"Slides: {rows['slide_id'].nunique() if not rows.empty else 0}")
    print(f"Encoders: {sorted(rows['feature_dir'].unique()) if not rows.empty else []}")

    if args.projection_checkpoint is None:
        print()
        print("No --projection-checkpoint provided. This run only inspected the manifest.")
        print("After Stage 1, pass the trained ProjectionHead checkpoint to build Beacon:")
        print(
            "python code/modules/beacon.py "
            "--projection-checkpoint path/to/stage1_projection.pt --fold 1"
        )
        return

    beacon, summary, input_dims = build_static_beacon_from_checkpoint(
        manifest_path=args.manifest,
        checkpoint_path=args.projection_checkpoint,
        fold=args.fold,
        target_dim=args.target_dim,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        dropout=args.dropout,
        normalize_beacon=args.normalize_beacon,
        strict_load=args.strict_load,
    )
    metadata = {
        "stage": "stage2_static_beacon",
        "manifest": str(args.manifest),
        "projection_checkpoint": str(args.projection_checkpoint),
        "fold": int(args.fold),
        "split": args.split,
        "target_dim": int(args.target_dim),
        "device": args.device,
        "normalize_beacon": bool(args.normalize_beacon),
        "input_dims": input_dims,
        "policy": "ProjectionHead frozen; Beacon built from train split only.",
    }
    save_beacon(beacon, summary, args.output_dir, args.fold, metadata=metadata)
    print(f"Saved Beacon: {args.output_dir / f'fold_{args.fold}_beacon.pt'}")
    print(f"Saved summary: {args.output_dir / f'fold_{args.fold}_beacon_summary.csv'}")


if __name__ == "__main__":
    main()
