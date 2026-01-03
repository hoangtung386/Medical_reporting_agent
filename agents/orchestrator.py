"""
Agent 3: Knowledge Fusion & Orchestrator (LLM-based).
Plans workflows and routes data to specialist agents.
"""
from .base import BaseAgent
from typing import Dict, Any, List
import os
import json

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

class OrchestratorAgent(BaseAgent):
    """
    Agent 3: Knowledge Fusion & Orchestrator (LLM-based).
    Plans workflows and routes data to specialist agents.
    """
    def __init__(self, specialists: Dict[str, BaseAgent]):
        super().__init__(name="Agent 3: Orchestrator")
        self.specialists = specialists
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if Anthropic and api_key:
            print(f"[{self.name}] Initializing Anthropic client...")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-sonnet-20240229"
        else:
            print(f"[{self.name}] WARNING: Anthropic client not available (missing key or lib). Using mock planner.")
            self.client = None

    def forward(self, vision_features: Any, seg_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decides which specialists to call based on initial vision and segmentation data.
        """
        print(f"[{self.name}] Planning workflow based on input...")
        
        # 1. Summarize input for LLM
        # vision_features is a tensor/array, we can't pass it directly. 
        # In a real system, we'd pass a summary or vision-to-language description.
        seg_summary = str(seg_output.get("metrics", {}))
        
        # 2. Generate Plan
        if self.client:
            # Real LLM call
            prompt = f"""
            Based on the following segmentation metrics from a medical scan, decide which specialists to activate.
            Metrics: {seg_summary}
            
            Available Specialists:
            - anatomy: detailed anatomical analysis
            - pathology: lesion characterization
            - measurement: precise quantification
            - rag: retrieve guidelines
            
            Return ONLY a JSON list of strings, e.g., ["anatomy", "measurement"].
            """
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                plan_str = message.content[0].text
                # Simple parsing logic - in production use robust JSON parser
                # plan = json.loads(plan_str) # Skipping for safety in this skeleton
                plan = ["anatomy", "pathology", "measurement", "rag"] # Fallback to full pipeline for demo
            except Exception as e:
                print(f"[{self.name}] Error in planning: {e}. Falling back to default.")
                plan = ["anatomy", "pathology", "measurement", "rag"]
        else:
             # Mock plan
             plan = ["anatomy", "pathology", "measurement", "rag"]
        
        print(f"[{self.name}] Activated Agents: {plan}")

        # 3. Execute Plan
        results = {}
        for agent_key in plan:
            if agent_key in self.specialists:
                agent = self.specialists[agent_key]
                print(f"[{self.name}] Routing to {agent.name}...")
                results[agent_key] = agent.forward(vision_features, seg_output)
            else:
                print(f"[{self.name}] Warning: Specialist '{agent_key}' not found.")
                
        return results
