import numpy as np
from agents.agent_1_vision.vision_encoder import VisionEncoderAgent
from agents.agent_2_segmentation.segmentation import SegmentationSpecialistAgent
from agents.agent_3_orchestrator.orchestrator import OrchestratorAgent
from agents.agent_4_anatomy.specialist import AnatomySpecialistAgent
from agents.agent_5_pathology.specialist import PathologySpecialistAgent
from agents.agent_6_measurement.quantifier import MeasurementQuantifierAgent
from agents.agent_7_rag.retriever import RAGSpecialistAgent
from agents.agent_8_report_gen.generator import ReportGeneratorAgent
from agents.agent_9_validator.validator import ClinicalValidatorAgent

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
