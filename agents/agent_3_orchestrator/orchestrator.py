"""
Agent 3: Knowledge Fusion & Orchestrator (LLM-based).
Plans workflows and routes data to specialist agents.
"""
from ..base import BaseAgent
from typing import Dict, Any, List
import os
import json

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

class OrchestratorAgent(BaseAgent):
    """
    Agent 3: Knowledge Fusion & Orchestrator (LLM-based).
    Plans workflows and routes data to specialist agents.
    Now using local LLM (GPT-OSS-20B) instead of API.
    """
    def __init__(self, specialists: Dict[str, BaseAgent], device="cpu"):
        super().__init__(name="Agent 3: Orchestrator")
        self.specialists = specialists
        
        # Determine device
        if torch and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None

        if AutoModelForCausalLM and torch:
            print(f"[{self.name}] Initializing Local LLM (openai/gpt-oss-20b)...")
            try:
                model_id = "openai/gpt-oss-20b"
                # In a real scenario, ensure this model exists or use a fallback. 
                # Since user requested this specific ID, we try to load it.
                # Note: This load might fail if model requires auth or is too large for memory.
                # We wrap in try-except to allow falling back to mock if weights aren't present.
                
                # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                # self.model = AutoModelForCausalLM.from_pretrained(model_id)
                # self.model.to(self.device)
                
                print(f"[{self.name}] Model loading skeleton ready. (Uncomment to run with real weights)")
                # For demo purposes, we will treat 'tokenizer' as None implies 'use mock'
                
            except Exception as e:
                print(f"[{self.name}] Warning: Could not init model 'openai/gpt-oss-20b': {e}")
        else:
            print(f"[{self.name}] WARNING: Transformers/Torch not installed. Using mock planner.")

    def forward(self, vision_features: Any, seg_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decides which specialists to call based on initial vision and segmentation data.
        """
        print(f"[{self.name}] Planning workflow based on input...")
        
        seg_summary = str(seg_output.get("metrics", {}))
        
        plan = ["anatomy", "pathology", "measurement", "rag"] # Default Fallback

        if self.model and self.tokenizer:
            # Real LLM call using Transformers
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
            
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=40)
                
                # Decode output
                generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                
                # Parse JSON from text (Mocking parser logic here)
                # plan = json.loads(generated_text)
                print(f"[{self.name}] LLM Output: {generated_text}")
                
            except Exception as e:
                print(f"[{self.name}] Error in generation: {e}")
        
        else:
             # Mock plan for skeleton
             pass
        
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
