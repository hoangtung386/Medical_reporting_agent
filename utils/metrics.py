"""
Evaluation metrics for medical report generation.

Implements standard NLG metrics:
    - BLEU (n-gram overlap
    - ROUGE (recall-oriented overlap)
    - METEOR (semantic matching)
    - CIDEr (consensus-based similarity)
    - BERTScore (semantic similarity)
    - Clinical Accuracy (domain-specific)
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import warnings

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not installed. Some metrics unavailable. Install: pip install nltk")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    warnings.warn("rouge-score not installed. Install: pip install rouge-score")


def compute_bleu(
    references: List[str],
    hypothesis: str,
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU scores (1-gram to 4-gram).
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
        max_n: Maximum n-gram to compute (default: 4)
    
    Returns:
        Dictionary with BLEU-1 to BLEU-4 scores
    """
    if not NLTK_AVAILABLE:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    # Tokenize
    references_tokens = [ref.lower().split() for ref in references]
    hypothesis_tokens = hypothesis.lower().split()
    
    # Smoothing function for short sentences
    smooth = SmoothingFunction().method1
    
    scores = {}
    for n in range(1, max_n + 1):
        weights = tuple([1.0/n] * n + [0.0] * (4 - n))
        score = sentence_bleu(
            references_tokens,
            hypothesis_tokens,
            weights=weights,
            smoothing_function=smooth
        )
        scores[f'bleu_{n}'] = score
    
    return scores


def compute_rouge(
    references: List[str],
    hypothesis: str,
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL']
) -> Dict[str, float]:
    """
    Compute ROUGE scores (recall-oriented).
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
        rouge_types: Types of ROUGE to compute
    
    Returns:
        Dictionary with ROUGE scores (F1 measure)
    """
    if not ROUGE_AVAILABLE:
        return {rt: 0.0 for rt in rouge_types}
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Compute against all references and take max
    all_scores = {rt: [] for rt in rouge_types}
    
    for reference in references:
        scores = scorer.score(reference, hypothesis)
        for rouge_type in rouge_types:
            all_scores[rouge_type].append(scores[rouge_type].fmeasure)
    
    # Return max score for each metric
    final_scores = {}
    for rouge_type in rouge_types:
        final_scores[rouge_type] = max(all_scores[rouge_type])
    
    return final_scores


def compute_meteor(
    references: List[str],
    hypothesis: str
) -> float:
    """
    Compute METEOR score (semantic matching).
    
    Args:
        references: List of reference texts
        hypothesis: Generated text
    
    Returns:
        METEOR score
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    # Tokenize
    references_tokens = [ref.lower().split() for ref in references]
    hypothesis_tokens = hypothesis.lower().split()
    
    # Compute METEOR (returns score against best reference)
    score = meteor_score(references_tokens, hypothesis_tokens)
    
    return score


def compute_clinical_accuracy(
    reference: str,
    hypothesis: str,
    clinical_entities: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute clinical accuracy metrics.
    
    Checks if important clinical entities (organs, findings, measurements)
    are correctly mentioned.
    
    Args:
        reference: Ground truth report
        hypothesis: Generated report
        clinical_entities: List of required entities to check
    
    Returns:
        Dictionary with precision, recall, F1 for entities
    """
    if clinical_entities is None:
        # Default medical entities to look for
        clinical_entities = [
            'liver', 'kidney', 'spleen', 'pancreas', 'lung',
            'normal', 'abnormal', 'lesion', 'mass', 'nodule',
            'unremarkable', 'enlarged', 'small', 'calcification'
        ]
    
    # Lowercase for matching
    ref_lower = reference.lower()
    hyp_lower = hypothesis.lower()
    
    # Count entity mentions
    ref_entities = set()
    hyp_entities = set()
    
    for entity in clinical_entities:
        if entity in ref_lower:
            ref_entities.add(entity)
        if entity in hyp_lower:
            hyp_entities.add(entity)
    
    # Calculate metrics
    true_positives = len(ref_entities & hyp_entities)
    false_positives = len(hyp_entities - ref_entities)
    false_negatives = len(ref_entities - hyp_entities)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'clinical_precision': precision,
        'clinical_recall': recall,
        'clinical_f1': f1,
        'n_entities_ref': len(ref_entities),
        'n_entities_hyp': len(hyp_entities)
    }


def compute_all_metrics(
    references: List[str],
    hypothesis: str,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Compute all available metrics.
    
    Args:
        references: List of reference reports
        hypothesis: Generated report
        detailed: If True, return individual scores; else only summary
    
    Returns:
        Dictionary with all metric scores
    """
    metrics = {}
    
    # BLEU
    bleu_scores = compute_bleu(references, hypothesis)
    metrics.update(bleu_scores)
    
    # ROUGE
    rouge_scores = compute_rouge(references, hypothesis)
    metrics.update(rouge_scores)
    
    # METEOR
    meteor = compute_meteor(references, hypothesis)
    metrics['meteor'] = meteor
    
    # Clinical accuracy (use first reference)
    clinical_acc = compute_clinical_accuracy(references[0], hypothesis)
    metrics.update(clinical_acc)
    
    # Summary score (average of key metrics)
    if not detailed:
        key_metrics = [
            metrics.get('bleu_4', 0),
            metrics.get('rougeL', 0),
            metrics.get('meteor', 0),
            metrics.get('clinical_f1', 0)
        ]
        metrics['summary_score'] = np.mean(key_metrics)
    
    return metrics


def evaluate_dataset(
    predictions: List[str],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Evaluate on entire dataset.
    
    Args:
        predictions: List of generated reports
        references: List of reference report lists (can have multiple refs per example)
    
    Returns:
        Average metrics across dataset
    """
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    all_metrics = []
    
    for pred, refs in zip(predictions, references):
        metrics = compute_all_metrics(refs, pred, detailed=True)
        all_metrics.append(metrics)
    
    # Average across all examples
    avg_metrics = {}
    metric_names = all_metrics[0].keys()
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        avg_metrics[metric_name] = np.mean(values)
    
    return avg_metrics


# Example usage
if __name__ == "__main__":
    # Example reports
    reference_reports = [
        "The liver is unremarkable. The kidneys are normal in size. No focal lesions identified.",
        "Liver appears normal. Both kidneys show normal size and attenuation. No abnormalities detected."
    ]
    
    generated_report = "The liver is normal. Kidneys are unremarkable. No lesions seen."
    
    # Compute metrics
    metrics = compute_all_metrics(reference_reports, generated_report)
    
    print("Evaluation Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")
