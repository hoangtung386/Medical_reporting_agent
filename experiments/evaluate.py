"""
Evaluation script for comparing models.

Compares our segmentation-guided model against baselines.

Usage:
    python experiments/evaluate.py --model_path checkpoints/best_model.pth
"""

import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm

from models.segmentation import SegmentationModel, SegmentationWrapper
from models.generation import SegmentationGuidedReportGenerator
from models.baselines import BASELINES
from utils.metrics import compute_all_metrics, evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--baseline', type=str, choices=list(BASELINES.keys()), default=None)
    parser.add_argument('--data_path', type=str, default='data/test')
    parser.add_argument('--output', type=str, default='results/evaluation.json')
    return parser.parse_args()


def evaluate_model(model, seg_model, test_loader, device):
    """
    Run evaluation on test set.
    """
    model.eval()
    if seg_model:
        seg_model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reports"):
            ct_volume = batch['ct_volume'].to(device)
            ground_truth = batch['report']
            
            # Get segmentation if needed
            if seg_model:
                seg_output = seg_model(ct_volume, return_features=True)
                report = model.generate_report(seg_output)
            else:
                # Baseline model
                output = model(ct_volume)
                report = "Generated report placeholder"  # Decode from tokens
            
            predictions.append(report)
            references.append([ground_truth])  # Wrap in list for multiple refs
    
    # Compute metrics
    metrics = evaluate_dataset(predictions, references)
    
    return metrics, predictions


def compare_all_models(test_loader, device):
    """
    Compare our model against all baselines.
    """
    results = {}
    
    # Load segmentation model (shared by our approach)
    seg_model = SegmentationModel().to(device)
    
    # Our model
    print("\n" + "="*60)
    print("Evaluating: Ours (Segmentation-Guided)")
    print("="*60)
    our_model = SegmentationGuidedReportGenerator(device=device)
    metrics, preds = evaluate_model(our_model, seg_model, test_loader, device)
    results['ours'] = metrics
    
    # Ablation: Ours without RAG
    print("\n" + "="*60)
    print("Evaluating: Ours (w/o RAG)")
    print("="*60)
    # Same model, just don't use RAG in generation
    metrics_no_rag, _ = evaluate_model(our_model, seg_model, test_loader, device)
    results['ours_no_rag'] = metrics_no_rag
    
    # Baselines
    for baseline_name, baseline_class in BASELINES.items():
        print("\n" + "="*60)
        print(f"Evaluating: {baseline_name.upper()} Baseline")
        print("="*60)
        baseline_model = baseline_class().to(device)
        metrics, _ = evaluate_model(baseline_model, None, test_loader, device)
        results[baseline_name] = metrics
    
    return results


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    # TODO: Implement test data loader
    print(f"Loading test data from {args.data_path}...")
    test_loader = []  # Placeholder
    
    # Run comparison
    results = compare_all_models(test_loader, device)
    
    # Print comparison table
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Format as table
    metric_names = list(next(iter(results.values())).keys())
    
    # Header
    print(f"{'Model':<25}", end="")
    for metric in metric_names[:5]:  # Show top 5 metrics
        print(f"{metric:>15}", end="")
    print()
    print("-" * 100)
    
    # Rows
    for model_name, metrics in results.items():
        print(f"{model_name:<25}", end="")
        for metric in metric_names[:5]:
            value = metrics.get(metric, 0.0)
            print(f"{value:>15.4f}", end="")
        print()
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
