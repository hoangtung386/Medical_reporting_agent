"""
Simple demonstration of the medical report generation pipeline.

This is the NEW, SIMPLIFIED main.py - replacing the over-engineered 9-agent version.

For research: Focus is on the NOVEL contribution (segmentation-guided generation),
not on complex orchestration.
"""

import numpy as np
import torch
import warnings

# Core models
from models.segmentation import SegmentationModel, SegmentationWrapper
from models.generation import SegmentationGuidedReportGenerator

# Utilities
from utils.measurements import calculate_volumes, format_measurements_for_report
from utils.rag import GuidelineRetriever

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """
    End-to-end pipeline demo: CT scan → Segmentation → Report
    
    Pipeline stages:
        1. Load CT volume
        2. 3D Segmentation (SwinUNETR)
        3. Calculate measurements (deterministic)
        4. Generate report (MedGemma + segmentation guidance)
        5. Optional: Retrieve guidelines (RAG)
    """
    
    print("=" * 70)
    print("MEDICAL REPORT GENERATION SYSTEM")
    print("Vision-Guided Report Generation via Segmentation-Aware Attention")
    print("=" * 70)
    
    # ==================== STAGE 1: Initialize Models ====================
    print("\n[1/5] Initializing models...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    # Segmentation model (combines Agent 1 + 2)
    seg_model = SegmentationModel(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=25,  # 25 organs in AbdomenAtlas
        pretrained_path=None  # Add path to SuPreM weights if available
    )
    seg_wrapper = SegmentationWrapper(seg_model, device=device)
    print("    ✓ Segmentation model loaded")
    
    # Report generator (Agent 8 - the CORE novelty)
    report_gen = SegmentationGuidedReportGenerator(
        model_name="google/medgemma-2b",
        use_lora=True,
        device=device
    )
    print("    ✓ Report generator loaded")
    
    # Optional: RAG for guidelines (Agent 7 - optional)
    use_rag = False  # Set to True for ablation study
    if use_rag:
        rag_retriever = GuidelineRetriever(db_path="./data/rag_db")
        print("    ✓ RAG retriever loaded")
    else:
        rag_retriever = None
        print("    ○ RAG retriever disabled (ablation mode)")
    
    # ==================== STAGE 2: Load Input Data ====================
    print("\n[2/5] Loading CT volume...")
    
    # Mock CT volume (replace with real data loader)
    ct_volume = torch.randn(1, 1, 96, 96, 96)  # [B, C, D, H, W]
    print(f"    CT shape: {ct_volume.shape}")
    print(f"    Spacing: (1.5, 1.0, 1.0) mm")
    
    # ==================== STAGE 3: Segmentation ====================
    print("\n[3/5] Running 3D segmentation...")
    
    seg_output = seg_wrapper.predict(
        ct_volume,
        return_features=True  # Need features for report generation
    )
    
    print(f"    ✓ Segmentation complete")
    print(f"    Detected organs: {seg_output['measurements']['num_organs_detected']}")
    
    # Display volumes
    volumes = seg_output['measurements']['volumes_mm3']
    if volumes:
        print("    Top 3 organs by volume:")
        sorted_organs = sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:3]
        for organ, vol in sorted_organs:
            print(f"      - {organ}: {vol/1000:.1f} cm³")
    
    # ==================== STAGE 4: Retrieve Guidelines (Optional) ====================
    rag_context = None
    if use_rag and rag_retriever:
        print("\n[4/5] Retrieving clinical guidelines...")
        
        # Query based on detected pathology (simplified)
        query = "abdominal ct findings interpretation"
        guidelines = rag_retriever.query(query, n_results=2)
        
        if guidelines:
            rag_context = "\n".join([g['document'] for g in guidelines])
            print(f"    ✓ Retrieved {len(guidelines)} relevant guidelines")
    else:
        print("\n[4/5] Skipping guideline retrieval (RAG disabled)")
    
    # ==================== STAGE 5: Generate Report ====================
    print("\n[5/5] Generating radiology report...")
    
    # Clinical indication (optional user input)
    clinical_indication = "Routine abdominal CT without contrast"
    
    # Generate report using segmentation-guided model
    report = report_gen.generate_report(
        seg_output=seg_output,
        clinical_indication=clinical_indication,
        rag_context=rag_context
    )
    
    print("    ✓ Report generated")
    
    # ==================== DISPLAY RESULTS ====================
    print("\n" + "=" * 70)
    print("GENERATED RADIOLOGY REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    
    # ==================== MEASUREMENTS (for validation) ====================
    print("\nQUANTITATIVE MEASUREMENTS:")
    print("-" * 70)
    measurement_text = format_measurements_for_report(seg_output['measurements'])
    print(measurement_text)
    print("-" * 70)
    
    print("\n✅ Pipeline complete!")
    print(f"\nPipeline summary:")
    print(f"  - Segmentation: {seg_output['measurements']['num_organs_detected']} organs detected")
    print(f"  - RAG: {'Enabled' if use_rag else 'Disabled'}")
    print(f"  - Report length: {len(report.split())} words")
    
    return {
        'segmentation': seg_output,
        'report': report,
        'measurements': seg_output['measurements']
    }


def inference_single_case(
    ct_path: str,
    output_path: str,
    clinical_indication: str = None,
    use_rag: bool = False
):
    """
    Run inference on a single CT scan from file.
    
    Args:
        ct_path: Path to CT scan (NIfTI format)
        output_path: Path to save report
        clinical_indication: Optional clinical context
        use_rag: Whether to use RAG for guidelines
    """
    import nibabel as nib
    
    # Load CT
    ct_nifti = nib.load(ct_path)
    ct_volume = torch.from_numpy(ct_nifti.get_fdata()).float()
    ct_volume = ct_volume.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    # Run pipeline (same as main() but with real data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seg_model = SegmentationModel()
    seg_wrapper = SegmentationWrapper(seg_model, device=device)
    
    report_gen = SegmentationGuidedReportGenerator(device=device)
    
    # Segment
    seg_output = seg_wrapper.predict(ct_volume, return_features=True)
    
    # Generate
    report = report_gen.generate_report(
        seg_output=seg_output,
        clinical_indication=clinical_indication
    )
    
    # Save
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {output_path}")
    return report


if __name__ == "__main__":
    # Run demo
    results = main()
    
    # Optional: Uncomment to run on real data
    # inference_single_case(
    #     ct_path="data/sample_ct.nii.gz",
    #     output_path="output/report.txt",
    #     clinical_indication="Routine screening"
    # )
