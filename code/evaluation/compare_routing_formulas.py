"""Compare attribution routing formulas on one saved GME evaluation folder.

This is an offline diagnostic. It reuses the saved attribution and signed
pairwise interactions, so no model inference or retraining is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage", default="inner_selection")
    parser.add_argument("--lambda", dest="interaction_lambda", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=2)
    return parser.parse_args()


def positive_topk(interactions: pd.DataFrame, names: list[str], topk: int) -> pd.DataFrame:
    rows = []
    for slide_id, group in interactions.groupby("slide_id", sort=False):
        matrix = np.zeros((len(names), len(names)), dtype=float)
        for row in group.itertuples(index=False):
            i = names.index(row.encoder_i)
            j = names.index(row.encoder_j)
            matrix[i, j] = matrix[j, i] = float(row.interaction)
        positive = np.maximum(matrix, 0.0)
        np.fill_diagonal(positive, 0.0)
        k = min(topk, max(len(names) - 1, 1))
        summary = np.sort(positive, axis=1)[:, -k:].mean(axis=1)
        rows.extend(
            {"slide_id": slide_id, "encoder": name, "interaction_topk_mean": float(value)}
            for name, value in zip(names, summary)
        )
    return pd.DataFrame(rows)


def routing_weights(scores: np.ndarray, minimum: float, maximum: float) -> tuple[np.ndarray, float, float]:
    normalized = np.clip((scores - minimum) / max(maximum - minimum, 1e-8), 0.0, 1.0)
    score_range = float(normalized.max() - normalized.min())
    tau = max(0.3, 0.6 / (1.0 + score_range))
    logits = normalized / tau
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, tau, score_range


def main() -> None:
    args = parse_args()
    fold_dirs = sorted(path for path in args.run_dir.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directory found under {args.run_dir}")

    for fold_dir in fold_dirs:
        weights_path = fold_dir / f"{args.stage}_routing_weights.csv"
        interactions_path = fold_dir / f"{args.stage}_pairwise_interactions.csv"
        stats_path = fold_dir / "routing_stats.json"
        if not weights_path.exists() or not interactions_path.exists():
            continue

        saved = pd.read_csv(weights_path)
        interactions = pd.read_csv(interactions_path)
        names = sorted(saved["encoder"].unique())
        topk = positive_topk(interactions, names, args.topk)
        data = saved[["slide_id", "encoder", "attribution"]].merge(
            topk, on=["slide_id", "encoder"], how="left", validate="one_to_one"
        )
        if args.interaction_lambda is None:
            # This run was the direct-addition experiment, whose config is
            # recorded in the parent directory when available.
            config_path = args.run_dir / "config.json"
            interaction_lambda = 0.1
            if config_path.exists():
                try:
                    import json

                    interaction_lambda = float(json.loads(config_path.read_text(encoding="utf-8"))["interaction_lambda"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
        else:
            interaction_lambda = float(args.interaction_lambda)

        interaction_scale = float(data["interaction_topk_mean"].std(ddof=0))
        if interaction_scale < 1e-8:
            interaction_scale = 1e-8

        stats = None
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        checkpoint_path = fold_dir / "best_gme_model.pt"
        if stats is None and checkpoint_path.exists():
            # The checkpoint stores the train-only routing statistics used to
            # produce the saved weights. Reuse them for an apples-to-apples
            # comparison instead of recomputing min/max on inner validation.
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            stats = checkpoint.get("router_stats") or checkpoint.get("routing_score_stats")
        if stats is not None:
            minimum = float(stats["attribution_min"])
            maximum = float(stats["attribution_max"])
        else:
            minimum = float(data["attribution"].min())
            maximum = float(data["attribution"].max())

        output_rows = []
        for slide_id, group in data.groupby("slide_id", sort=False):
            group = group.set_index("encoder").loc[names]
            attribution = group["attribution"].to_numpy(dtype=float)
            interaction = group["interaction_topk_mean"].to_numpy(dtype=float)
            score_none = attribution
            score_add = attribution + interaction_lambda * interaction
            score_multiplicative = attribution * (
                1.0 + interaction_lambda * np.tanh(interaction / interaction_scale)
            )

            weights_none, tau_none, range_none = routing_weights(score_none, minimum, maximum)
            weights_add, tau_add, range_add = routing_weights(score_add, minimum, maximum)
            weights_multiplicative, tau_multiplicative, range_multiplicative = routing_weights(
                score_multiplicative, minimum, maximum
            )
            for index, encoder in enumerate(names):
                output_rows.append(
                    {
                        "slide_id": slide_id,
                        "encoder": encoder,
                        "attribution": attribution[index],
                        "interaction_topk_mean": interaction[index],
                        "interaction_scale_std": interaction_scale,
                        "score_none": score_none[index],
                        "score_direct_add": score_add[index],
                        "score_current_multiplicative": score_multiplicative[index],
                        "weight_none": weights_none[index],
                        "weight_direct_add": weights_add[index],
                        "weight_current_multiplicative": weights_multiplicative[index],
                        "delta_score_current_vs_add": score_multiplicative[index] - score_add[index],
                        "delta_weight_current_vs_add": weights_multiplicative[index] - weights_add[index],
                        "delta_weight_current_vs_none": weights_multiplicative[index] - weights_none[index],
                        "tau_none": tau_none,
                        "tau_direct_add": tau_add,
                        "tau_current_multiplicative": tau_multiplicative,
                        "score_range_none": range_none,
                        "score_range_direct_add": range_add,
                        "score_range_current_multiplicative": range_multiplicative,
                    }
                )

        result = pd.DataFrame(output_rows)
        result.to_csv(
            fold_dir / "routing_formula_comparison.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.9f",
        )
        summary = pd.DataFrame(
            [
                {
                    "fold": fold_dir.name,
                    "interaction_lambda": interaction_lambda,
                    "interaction_topk": args.topk,
                    "interaction_scale_std": interaction_scale,
                    "mean_abs_delta_weight_current_vs_add": result["delta_weight_current_vs_add"].abs().mean(),
                    "max_abs_delta_weight_current_vs_add": result["delta_weight_current_vs_add"].abs().max(),
                    "mean_abs_delta_weight_current_vs_none": result["delta_weight_current_vs_none"].abs().mean(),
                    "max_abs_delta_weight_current_vs_none": result["delta_weight_current_vs_none"].abs().max(),
                    "mean_abs_delta_score_current_vs_add": result["delta_score_current_vs_add"].abs().mean(),
                    "max_abs_delta_score_current_vs_add": result["delta_score_current_vs_add"].abs().max(),
                    "routing_attribution_min": minimum,
                    "routing_attribution_max": maximum,
                }
            ]
        )
        summary.to_csv(
            fold_dir / "routing_formula_comparison_summary.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.9f",
        )
        print(f"Wrote {fold_dir / 'routing_formula_comparison.csv'}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
