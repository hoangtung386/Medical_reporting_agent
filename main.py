"""End-to-end demo of the medical report generation pipeline.

Pipeline stages:
    1. Initialise models (segmentation + report generator)
    2. Load CT volume
    3. Run 3D segmentation
    4. Calculate deterministic measurements
    5. Generate radiology report
"""

import logging
import warnings
from typing import Any, Dict, Optional

import torch

from models.generation import SegmentationGuidedReportGenerator
from models.segmentation import SegmentationModel, SegmentationWrapper
from utils.measurements import calculate_volumes, format_measurements_for_report
from utils.rag import GuidelineRetriever

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main() -> Dict[str, Any]:
    """Run the full demo pipeline on mock data."""
    print("=" * 70)
    print("MEDICAL REPORT GENERATION SYSTEM")
    print("Vision-Guided Report Generation via Segmentation-Aware Attention")
    print("=" * 70)

    # --- Stage 1: models ---
    print("\n[1/5] Initialising models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    seg_model = SegmentationModel(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=25,
        pretrained_path=None,
    )
    seg_wrapper = SegmentationWrapper(seg_model, device=device)
    print("    Segmentation model loaded")

    report_gen = SegmentationGuidedReportGenerator(
        model_name="google/medgemma-2b",
        use_lora=True,
        device=device,
    )
    print("    Report generator loaded")

    use_rag = False
    if use_rag:
        rag_retriever = GuidelineRetriever(db_path="./data/rag_db")
        print("    RAG retriever loaded")
    else:
        rag_retriever = None
        print("    RAG retriever disabled (ablation mode)")

    # --- Stage 2: input data ---
    print("\n[2/5] Loading CT volume...")
    ct_volume = torch.randn(1, 1, 96, 96, 96)
    print(f"    CT shape: {ct_volume.shape}")
    print("    Spacing: (1.5, 1.0, 1.0) mm")

    # --- Stage 3: segmentation ---
    print("\n[3/5] Running 3D segmentation...")
    seg_output = seg_wrapper.predict(ct_volume, return_features=True)
    print("    Segmentation complete")
    print(
        f"    Detected organs: "
        f"{seg_output['measurements']['num_organs_detected']}"
    )

    volumes = seg_output["measurements"]["volumes_mm3"]
    if volumes:
        print("    Top 3 organs by volume:")
        sorted_organs = sorted(
            volumes.items(), key=lambda x: x[1], reverse=True
        )[:3]
        for organ, vol in sorted_organs:
            print(f"      - {organ}: {vol / 1000:.1f} cm3")

    # --- Stage 4: guidelines (optional) ---
    rag_context: Optional[str] = None
    if use_rag and rag_retriever:
        print("\n[4/5] Retrieving clinical guidelines...")
        guidelines = rag_retriever.query(
            "abdominal ct findings interpretation", n_results=2
        )
        if guidelines:
            rag_context = "\n".join(g["document"] for g in guidelines)
            print(f"    Retrieved {len(guidelines)} guidelines")
    else:
        print("\n[4/5] Skipping guideline retrieval (RAG disabled)")

    # --- Stage 5: report generation ---
    print("\n[5/5] Generating radiology report...")
    report = report_gen.generate_report(
        seg_output=seg_output,
        clinical_indication="Routine abdominal CT without contrast",
        rag_context=rag_context,
    )
    print("    Report generated")

    # --- Display ---
    print("\n" + "=" * 70)
    print("GENERATED RADIOLOGY REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)

    print("\nQUANTITATIVE MEASUREMENTS:")
    print("-" * 70)
    print(format_measurements_for_report(seg_output["measurements"]))
    print("-" * 70)

    print("\nPipeline complete!")
    n_organs = seg_output["measurements"]["num_organs_detected"]
    print(f"  - Segmentation: {n_organs} organs detected")
    print(f"  - RAG: {'Enabled' if use_rag else 'Disabled'}")
    print(f"  - Report length: {len(report.split())} words")

    return {
        "segmentation": seg_output,
        "report": report,
        "measurements": seg_output["measurements"],
    }


def inference_single_case(
    ct_path: str,
    output_path: str,
    clinical_indication: Optional[str] = None,
    use_rag: bool = False,
) -> str:
    """Run inference on a single CT scan file.

    Args:
        ct_path: Path to NIfTI CT scan.
        output_path: Path to save the generated report.
        clinical_indication: Optional clinical context.
        use_rag: Whether to use RAG for guidelines.

    Returns:
        Generated report text.
    """
    import nibabel as nib

    ct_nifti = nib.load(ct_path)
    ct_volume = torch.from_numpy(ct_nifti.get_fdata()).float()
    ct_volume = ct_volume.unsqueeze(0).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = SegmentationModel()
    seg_wrapper = SegmentationWrapper(seg_model, device=device)
    report_gen = SegmentationGuidedReportGenerator(device=device)

    seg_output = seg_wrapper.predict(ct_volume, return_features=True)
    report = report_gen.generate_report(
        seg_output=seg_output,
        clinical_indication=clinical_indication,
    )

    with open(output_path, "w") as fh:
        fh.write(report)

    logger.info("Report saved to %s", output_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
