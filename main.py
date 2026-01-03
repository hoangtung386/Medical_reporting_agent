import numpy as np
from agents.vision_encoder import VisionEncoderAgent
from agents.segmentation import SegmentationSpecialistAgent
from agents.orchestrator import OrchestratorAgent
from agents.specialists import AnatomySpecialistAgent, PathologySpecialistAgent, MeasurementQuantifierAgent
from agents.rag_retrieval import RAGSpecialistAgent
from agents.report_gen import ReportGeneratorAgent
from agents.validator import ClinicalValidatorAgent

def main():
    print("🚀 Initializing AMMFS: Agentic Multi-Modal Foundation System...")
    
    # 1. Initialize Agents
    agent1_vision = VisionEncoderAgent()
    agent2_seg = SegmentationSpecialistAgent()
    
    specialists = {
        "anatomy": AnatomySpecialistAgent(),
        "pathology": PathologySpecialistAgent(),
        "measurement": MeasurementQuantifierAgent(),
        "rag": RAGSpecialistAgent()
    }
    
    agent3_orch = OrchestratorAgent(specialists)
    agent8_gen = ReportGeneratorAgent()
    agent9_val = ClinicalValidatorAgent()
    
    # 2. Load Input Data (Mock)
    print("\n--- [Step 0] Loading 3D CT Volume ---")
    ct_volume = np.random.rand(1, 256, 256, 256)
    print(f"Input loaded: {ct_volume.shape}")
    
    # 3. Execution Pipeline
    print("\n--- [Step 1] Global Vision Encoding ---")
    vision_features = agent1_vision.forward(ct_volume)
    
    print("\n--- [Step 2] Segmentation Specialist ---")
    seg_output = agent2_seg.forward(ct_volume, vision_features)
    
    print("\n--- [Step 3] Orchestrator Planning & Routing ---")
    # Orchestrator decides who to call and gathers results
    specialist_results = agent3_orch.forward(vision_features, seg_output)
    
    print("\n--- [Step 4] Report Generation ---")
    # Combine all data for the generator
    aggregated_data = {**specialist_results}
    report = agent8_gen.forward(aggregated_data)
    print("\nGenerated Report Preview:")
    print(report)
    
    print("\n--- [Step 5] Clinical Validation ---")
    errors = agent9_val.forward(report, seg_output["metrics"])
    
    if not errors:
        print("\n✅ Final Report Approved.")
    else:
        print("\n⚠️ Report requires manual review.")

if __name__ == "__main__":
    main()
