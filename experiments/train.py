"""Training script for medical report generation.

Trains the segmentation-guided report generator on CT-report pairs.

Usage::

    python -m experiments.train --config configs/abdomen_atlas_config.yaml
"""

from __future__ import annotations


import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.datasets.abdomen_atlas import AbdomenAtlasDataset, collate_fn
from models.baselines import BASELINES
from models.generation.medgemma import SegmentationGuidedReportGenerator
from models.generation.trainer import ReportGeneratorTrainer
from models.segmentation import SegmentationModel
from utils.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ------------------------------------------------------------------
# Config handling
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train report generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/abdomen_atlas_config.yaml",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINES.keys()),
        default=None,
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable W&B logging"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and normalise a YAML config file.

    Handles both flat configs (``train_config.yaml``) and nested ones
    (``abdomen_atlas_config.yaml``) by ensuring a ``training`` section
    always exists.
    """
    with open(config_path, "r") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    # Normalise: ensure training section exists
    if "training" not in config:
        config["training"] = {}

    _defaults = {
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "weight_decay": 0.01,
    }
    for key, default in _defaults.items():
        if key not in config["training"]:
            # Try root-level key first, then fallback to default
            config["training"][key] = config.pop(key, default)

    # eval_every / save_every
    for key in ("eval_every", "save_every"):
        if key not in config["training"]:
            config["training"][key] = config.pop(key, 2)

    # output_dir
    if "output_dir" not in config:
        config["output_dir"] = "checkpoints"

    return config


# ------------------------------------------------------------------
# Training & evaluation loops
# ------------------------------------------------------------------


def train_epoch(
    model: torch.nn.Module,
    seg_model: Optional[SegmentationModel],
    train_loader: DataLoader,
    trainer: ReportGeneratorTrainer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    if seg_model is not None:
        seg_model.eval()

    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch in progress:
        ct_volume = batch["ct_volume"].to(device)
        target_report = batch["report"]

        # Segmentation features (frozen)
        with torch.no_grad():
            seg_output = seg_model(ct_volume, return_features=True)

        seg_features = seg_output["features"]["bottleneck"]
        measurements = seg_output.get("measurements", {})

        # Language-modelling loss (first report in batch)
        loss = trainer.compute_loss(
            seg_features, measurements, target_report[0]
        )

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(len(train_loader), 1)


def evaluate(
    model: torch.nn.Module,
    seg_model: Optional[SegmentationModel],
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    if seg_model is not None:
        seg_model.eval()

    all_predictions: list[str] = []
    all_references: list[str] = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            ct_volume = batch["ct_volume"].to(device)
            reference_reports = batch["report"]

            seg_output = seg_model(ct_volume, return_features=True)
            generated = model.generate_report(seg_output)

            all_predictions.append(generated)
            all_references.extend(reference_reports)

    # Per-sample metrics, then average
    per_sample = [
        compute_all_metrics([ref], pred, detailed=True)
        for ref, pred in zip(all_references, all_predictions)
    ]

    if not per_sample:
        return {}

    metric_names = per_sample[0].keys()
    return {
        k: float(np.mean([m[k] for m in per_sample]))
        for k in metric_names
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """Entry-point for training."""
    args = parse_args()
    config = load_config(args.config)

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.get("wandb_project", "medical-report-gen"),
            config=config,
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    # --- Data ---
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get(
        "data_dir", "dataset_Abdomen_Atlas_3.0_mini_small"
    )
    csv_path = f"{data_dir}/data/AbdomenAtlas3.0MiniWithMeta.csv"

    logger.info("Loading dataset from %s ...", data_dir)
    train_dataset = AbdomenAtlasDataset(
        csv_path=csv_path,
        data_dir=f"{data_dir}/data",
        report_type=data_cfg.get("report_type", "narrative"),
        load_images=True,
    )
    val_dataset = AbdomenAtlasDataset(
        csv_path=csv_path,
        data_dir=f"{data_dir}/data",
        report_type=data_cfg.get("report_type", "narrative"),
        load_images=True,
    )

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --- Models ---
    if args.baseline:
        logger.info("Training baseline model: %s", args.baseline)
        model = BASELINES[args.baseline]()
        seg_model = None
    else:
        logger.info("Training segmentation-guided model")
        model_cfg = config.get("model", {})
        seg_weights = model_cfg.get("seg_weights_path") or model_cfg.get(
            "segmentation", {}
        ).get("pretrained")

        seg_model = SegmentationModel(
            pretrained_path=seg_weights,
        ).to(device)

        gen_cfg = model_cfg.get("generation", model_cfg)
        model = SegmentationGuidedReportGenerator(
            model_name=gen_cfg.get("name", "google/medgemma-2b"),
            use_lora=gen_cfg.get("use_lora", True),
            device=str(device),
        )

    trainer = ReportGeneratorTrainer(
        model,
        learning_rate=config["training"]["learning_rate"],
    )

    # --- Training loop ---
    best_metric = 0.0
    eval_every = config["training"].get("eval_every", 2)
    num_epochs = config["training"]["num_epochs"]

    for epoch in range(num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, num_epochs)

        train_loss = train_epoch(
            model, seg_model, train_loader, trainer, device, epoch
        )
        logger.info("Train loss: %.4f", train_loss)

        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, seg_model, val_loader, device)
            for k, v in metrics.items():
                logger.info("  %s: %.4f", k, v)

            if args.wandb and WANDB_AVAILABLE:
                wandb.log(
                    {"epoch": epoch, "train_loss": train_loss, **metrics}
                )

            clinical_f1 = metrics.get("clinical_f1", 0)
            if clinical_f1 > best_metric:
                best_metric = clinical_f1
                ckpt_path = (
                    Path(config["output_dir"]) / "best_model.pth"
                )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": (
                            trainer.optimizer.state_dict()
                        ),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )
                logger.info("Saved best model to %s", ckpt_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
