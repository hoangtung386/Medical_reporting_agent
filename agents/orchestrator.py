from .base import BaseAgent
from typing import Dict, Any, List

class OrchestratorAgent(BaseAgent):
    """
    Agent 3: Knowledge Fusion & Orchestrator (LLM-based).
    Plans workflows and routes data to specialist agents.
    """
    def __init__(self, specialists: Dict[str, BaseAgent]):
        super().__init__(name="Agent 3: Orchestrator")
        self.llm_model = "claude-sonnet-4-mock"
        self.specialists = specialists
        print(f"[{self.name}] Initializing Orchestrator with {len(specialists)} specialists...")

    def forward(self, vision_features: Any, seg_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decides which specialists to call based on initial vision and segmentation data.
        """
        print(f"[{self.name}] Planning workflow based on input...")
        
        # Mock LLM decision: "I see a lung nodule, better check anatomy, pathology, and measure it."
        plan = ["anatomy", "pathology", "measurement", "rag"]
        
        results = {}
        for agent_key in plan:
            if agent_key in self.specialists:
                agent = self.specialists[agent_key]
                print(f"[{self.name}] Routing to {agent.name}...")
                # In a real dynamic system, arguments would be tailored. 
                # Here we pass everything for the stub.
                results[agent_key] = agent.forward(vision_features, seg_output)
            else:
                print(f"[{self.name}] Warning: Specialist '{agent_key}' not found.")
                
        return results
