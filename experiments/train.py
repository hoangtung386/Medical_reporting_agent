"""
Training script for medical report generation.

Trains the segmentation-guided report generator on CT-report pairs.

Usage:
    python experiments/train.py --config configs/train_config.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import wandb

# Models
from models.segmentation import SegmentationModel
from models.generation import SegmentationGuidedReportGenerator
from models.baselines import BASELINES

# Utils
from utils.metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train report generator")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--baseline', type=str, choices=['lstm', 'transformer', 'simple_cnn_lstm', None], default=None)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model,
    seg_model,
    train_loader,
    optimizer,
    device,
    epoch,
    trainer=None
):
    """
    Train for one epoch.
    """
    model.train()
    seg_model.eval()  # Freeze segmentation model
    
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        ct_volume = batch['ct_volume'].to(device)
        target_report = batch['report']
        
        # Get segmentation features (no grad)
        with torch.no_grad():
            seg_output = seg_model(ct_volume, return_features=True)
        
        # Forward pass through report generator
        # We need a Trainer instance or call the loss method directly
        # For simplicity, we assume 'model' here is the trainer or we create one
        # Ideally, we should refactor main() to pass a trainer, but let's just 
        # use the method if we can, or instantiate the helper class.
        
        # NOTE: In main(), we probably want to wrap 'model' in ReportGeneratorTrainer
        # But for minimal changes, let's just do:
        
        # Get features
        seg_features = seg_output['features']['bottleneck']
        measurements = seg_output.get('measurements', {})
        
        # Compute loss
        # We need to access the trainer's compute_loss method. 
        # Since we modified main() to pass just 'model' (which is the nn.Module), 
        # let's create a temporary helper or assume we change main() too.
        # Let's change main() to create the trainer.
        
        loss = trainer.compute_loss(seg_features, measurements, target_report)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(
    model,
    seg_model,
    val_loader,
    device
):
    """
    Evaluate model on validation set.
    """
    model.eval()
    seg_model.eval()
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            ct_volume = batch['ct_volume'].to(device)
            reference_reports = batch['report']
            
            # Get segmentation
            seg_output = seg_model(ct_volume, return_features=True)
            
            # Generate report
            generated = model.generate_report(seg_output)
            
            all_predictions.append(generated)
            all_references.append(reference_reports)
    
    # Compute metrics
    metrics = compute_all_metrics(all_references, all_predictions)
    
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(project="medical-report-generation", config=config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    from data.datasets.loader import AbdomenAtlasDataset
    
    print(f"Loading dataset from {config['data']['data_dir']}...")
    train_dataset = AbdomenAtlasDataset(
        root_dir=config['data']['data_dir'],
        split='train'
    )
    val_dataset = AbdomenAtlasDataset(
        root_dir=config['data']['data_dir'],
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0 # Windows typically needs 0 for simple scripts
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Load models
    if args.baseline:
        print(f"Training baseline model: {args.baseline}")
        model = BASELINES[args.baseline]()
        seg_model = None
    else:
        print("Training novel segmentation-guided model")
        seg_model = SegmentationModel(
            pretrained_path=config.get('seg_weights_path')
        ).to(device)
        
        model = SegmentationGuidedReportGenerator(
            model_name=config['model']['name'],
            use_lora=config['model']['use_lora'],
            device=device
        )
    
    # Optimizer is now handled by ReportGeneratorTrainer
    
    # Trainer wrapper
    from models.generation.medgemma import ReportGeneratorTrainer
    trainer = ReportGeneratorTrainer(model, learning_rate=config['training']['learning_rate'])
    
    # Training loop
    best_metric = 0.0
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, seg_model, train_loader, trainer.optimizer, device, epoch, trainer)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['eval_every'] == 0:
            metrics = evaluate(model, seg_model, val_loader, device)
            print(f"Validation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Log to wandb
            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **metrics
                })
            
            # Save best model
            if metrics.get('clinical_f1', 0) > best_metric:
                best_metric = metrics['clinical_f1']
                checkpoint_path = Path(config['output_dir']) / 'best_model.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics
                }, checkpoint_path)
                print(f"✓ Saved best model to {checkpoint_path}")
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
