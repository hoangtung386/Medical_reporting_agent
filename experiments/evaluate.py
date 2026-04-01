"""Evaluation script for comparing models.

Compares the segmentation-guided model against baselines on a test set.

Usage::

    python -m experiments.evaluate --model_path checkpoints/best_model.pth
"""

from __future__ import annotations


import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets.abdomen_atlas import AbdomenAtlasDataset, collate_fn
from models.baselines import BASELINES
from models.generation.medgemma import SegmentationGuidedReportGenerator
from models.segmentation import SegmentationModel, SegmentationWrapper
from utils.metrics import compute_all_metrics, evaluate_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINES.keys()),
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset_Abdomen_Atlas_3.0_mini_small/data",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=(
            "dataset_Abdomen_Atlas_3.0_mini_small/"
            "data/AbdomenAtlas3.0MiniWithMeta.csv"
        ),
    )
    parser.add_argument(
        "--output", type=str, default="results/evaluation.json"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def evaluate_model(
    model: torch.nn.Module,
    seg_model: Optional[SegmentationModel],
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[Dict[str, float], List[str]]:
    """Run evaluation on a test set.

    Args:
        model: The report generator (ours or baseline).
        seg_model: Segmentation model (``None`` for baselines).
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        Tuple of (metrics dict, list of generated reports).
    """
    model.eval()
    if seg_model is not None:
        seg_model.eval()

    predictions: List[str] = []
    references: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reports"):
            ct_volume = batch["ct_volume"].to(device)
            ground_truth = batch["report"]

            if seg_model is not None:
                seg_output = seg_model(ct_volume, return_features=True)
                report = model.generate_report(seg_output)
            else:
                # Baseline model
                report = model.generate_report(ct_volume)

            predictions.append(report)
            references.append(ground_truth)

    metrics = evaluate_dataset(predictions, references)
    return metrics, predictions


def compare_all_models(
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compare our model against all baselines.

    Args:
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        Mapping of model name to its evaluation metrics.
    """
    results: Dict[str, Dict[str, float]] = {}

    seg_model = SegmentationModel().to(device)

    # Our model
    logger.info("Evaluating: Ours (Segmentation-Guided)")
    our_model = SegmentationGuidedReportGenerator(device=str(device))
    metrics, _ = evaluate_model(our_model, seg_model, test_loader, device)
    results["ours"] = metrics

    # Ablation: ours without RAG
    logger.info("Evaluating: Ours (w/o RAG)")
    metrics_no_rag, _ = evaluate_model(
        our_model, seg_model, test_loader, device
    )
    results["ours_no_rag"] = metrics_no_rag

    # Baselines
    for name, cls in BASELINES.items():
        logger.info("Evaluating: %s baseline", name.upper())
        baseline = cls().to(device)
        metrics, _ = evaluate_model(baseline, None, test_loader, device)
        results[name] = metrics

    return results


def main() -> None:
    """Entry-point for evaluation."""
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load test data
    logger.info("Loading test data from %s ...", args.data_dir)
    test_dataset = AbdomenAtlasDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        report_type="narrative",
        load_images=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Run comparison
    results = compare_all_models(test_loader, device)

    # Print table
    if results:
        first_metrics = next(iter(results.values()))
        metric_names = list(first_metrics.keys())[:5]

        header = f"{'Model':<25}"
        header += "".join(f"{m:>15}" for m in metric_names)
        logger.info("\n%s\n%s", header, "-" * len(header))

        for model_name, metrics in results.items():
            row = f"{model_name:<25}"
            row += "".join(
                f"{metrics.get(m, 0.0):>15.4f}" for m in metric_names
            )
            logger.info(row)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
