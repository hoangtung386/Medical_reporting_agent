from .base import BaseAgent
from typing import Any, Dict, List
import re

class ClinicalValidatorAgent(BaseAgent):
    """
    Agent 9: Clinical Validator.
    Cross-checks the generated report against quantitative data and masks.
    """
    def __init__(self):
        super().__init__(name="Agent 9: Clinical Validator")

    def forward(self, report: str, seg_metrics: Dict[str, Any]) -> List[str]:
        print(f"[{self.name}] Validating report consistency...")
        
        errors = []
        
        # 1. Check Laterality (Left/Right)
        # Simple heuristic: if report mentions "Left" but metrics say "Right" (if we had location in metrics)
        # For now, we look for contradictions within the report or vs mock data
        report_lower = report.lower()
        if "right" in report_lower and "left" in report_lower:
             # This might be valid (bilateral), but let's warn if it's a single lesion case
             pass 

        # 2. Check Measurements
        # Extract numbers from report: "3.2 cm"
        # Find all floats
        report_nums = [float(x) for x in re.findall(r"(\d+\.\d+)", report)]
        
        # Get ground truth from metrics
        # metrics["lesion_sizes"] = [3.2, 2.8]
        gt_sizes = seg_metrics.get("dimensions_cm", []) # [3.2, 2.8, 2.5] from specialist
        
        # Check if GT sizes appear in report
        for size in gt_sizes:
            if size > 0:
                # Allow small tolerance
                match = any(abs(rn - size) < 0.1 for rn in report_nums)
                if not match:
                    errors.append(f"Measurement {size} cm not found in report text.")
             
        if not errors:
            print(f"[{self.name}] No critical errors found.")
        else:
            print(f"[{self.name}] ERRORS FOUND: {errors}")
            
        return errors
