"""
Segmentation-Guided Medical Report Generator.

This is the CORE NOVELTY of the research:
    - Uses 3D segmentation features to guide report generation
    - Novel segmentation-aware cross-attention mechanism
    - Attends to anatomical regions when generating descriptions

Based on MedGemma-2B with custom attention layers.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import warnings

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not installed. Install with: pip install transformers")


class SegmentationAwareAttention(nn.Module):
    """
    NOVEL COMPONENT: Cross-attention between LLM hidden states and segmentation features.
    
    This allows the model to "look at" specific anatomical regions when generating
    descriptions (e.g., attending to liver when writing "liver: unremarkable").
    
    This is the key innovation over baseline report generators.
    """
    
    def __init__(
        self,
        llm_hidden_size: int = 2048,
        seg_feature_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = llm_hidden_size // num_heads
        
        # Project segmentation features to LLM dimension
        self.seg_projector = nn.Linear(seg_feature_size, llm_hidden_size)
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(llm_hidden_size)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_hidden_size * 4, llm_hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(llm_hidden_size)
        
    def forward(
        self, 
        llm_hidden_states: torch.Tensor,
        seg_features: torch.Tensor,
        return_attention_weights: bool = False
    ):
        """
        Args:
            llm_hidden_states: [batch, seq_len, llm_hidden_size]
            seg_features: [batch, num_regions, seg_feature_size]
            return_attention_weights: Return attention map for visualization
        
        Returns:
            conditioned_states: [batch, seq_len, llm_hidden_size]
            attention_weights: [batch, num_heads, seq_len, num_regions] (optional)
        """
        # Project segmentation features to LLM space
        seg_projected = self.seg_projector(seg_features)  # [B, num_regions, llm_hidden]
        
        # Cross-attention: Query from LLM, Key/Value from segmentation
        attn_output, attn_weights = self.cross_attn(
            query=llm_hidden_states,
            key=seg_projected,
            value=seg_projected,
            need_weights=return_attention_weights
        )
        
        # Residual connection + norm
        hidden_states = self.layer_norm(llm_hidden_states + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(hidden_states)
        output = self.ffn_norm(hidden_states + ffn_output)
        
        if return_attention_weights:
            return output, attn_weights
        return output


class SegmentationGuidedReportGenerator(nn.Module):
    """
    Medical Report Generator guided by 3D segmentation features.
    
    Architecture:
        1. Base LLM: MedGemma-2B (medical domain pre-trained)
        2. Segmentation Feature Encoder: Projects 3D features to sequence
        3. Cross-Attention Layers: Fuse segmentation info into generation
        4. Optional: RAG retrieval for clinical guidelines
    
    Training:
        - Phase 1: Train with frozen LLM, only train projection layers
        - Phase 2: LoRA fine-tuning of LLM layers
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        seg_feature_size: int = 768,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.seg_feature_size = seg_feature_size
        
        if not TRANSFORMERS_AVAILABLE:
            print("WARNING: Running in mock mode. Install transformers for real inference.")
            self.llm = None
            self.tokenizer = None
            self.llm_hidden_size = 2048
        else:
            # Load base model
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map=device
            )
            
            # Get hidden size
            self.llm_hidden_size = self.llm.config.hidden_size
            
            # Apply LoRA for efficient fine-tuning
            if use_lora:
                self._apply_lora(lora_r, lora_alpha)
            
            # Ensure padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Segmentation feature processing
        self.seg_feature_pooling = nn.AdaptiveAvgPool3d((4, 4, 4))  # Reduce spatial dims
        self.seg_feature_flatten = nn.Flatten()
        self.seg_to_sequence = nn.Linear(seg_feature_size * 64, self.llm_hidden_size * 8)
        
        # Cross-attention layers (NOVEL COMPONENT)
        self.cross_attention_layers = nn.ModuleList([
            SegmentationAwareAttention(
                llm_hidden_size=self.llm_hidden_size,
                seg_feature_size=self.llm_hidden_size  # After projection
            )
            for _ in range(3)  # Insert 3 cross-attention layers
        ])
        
        # Measurement conditioning (from Agent 6 calculations)
        self.measurement_encoder = nn.Sequential(
            nn.Linear(25, 128),  # 25 organ volumes
            nn.ReLU(),
            nn.Linear(128, self.llm_hidden_size)
        )
    
    def _apply_lora(self, r: int, alpha: int):
        """Apply LoRA for parameter-efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            print(f"Applied LoRA with r={r}, alpha={alpha}")
            print(f"Trainable parameters: {self.llm.print_trainable_parameters()}")
            
        except ImportError:
            warnings.warn("PEFT not installed. Running without LoRA. Install with: pip install peft")
    
    def encode_segmentation_features(
        self, 
        seg_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert 3D segmentation features to sequence representation.
        
        Args:
            seg_features: [batch, channels, D, H, W]
        
        Returns:
            feature_sequence: [batch, seq_len, hidden_size]
        """
        # Spatial pooling
        pooled = self.seg_feature_pooling(seg_features)  # [B, C, 4, 4, 4]
        flattened = self.seg_feature_flatten(pooled)  # [B, C*64]
        
        # Project to sequence
        seq = self.seg_to_sequence(flattened)  # [B, hidden*8]
        seq = seq.reshape(seq.size(0), 8, self.llm_hidden_size)  # [B, 8, hidden]
        
        return seq
    
    def encode_measurements(
        self,
        measurements: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Encode quantitative measurements as continuous embeddings.
        
        Args:
            measurements: Dict with 'volumes_mm3' for each organ
        
        Returns:
            measurement_embedding: [batch, hidden_size]
        """
        # Extract volumes for 25 organs (0 if not detected)
        volumes = measurements.get('volumes_mm3', {})
        volume_tensor = torch.zeros(25, device=self.device)
        
        for organ_id, volume in volumes.items():
            if organ_id < 25:
                volume_tensor[organ_id] = volume
        
        # Normalize (log scale)
        volume_tensor = torch.log1p(volume_tensor)
        
        # Encode
        embedding = self.measurement_encoder(volume_tensor.unsqueeze(0))
        
        return embedding
    
    def forward(
        self,
        seg_features: torch.Tensor,
        measurements: Dict[str, Any],
        prompt: Optional[str] = None,
        max_length: int = 512,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Generate medical report conditioned on segmentation.
        
        Args:
            seg_features: Encoder features from SegmentationModel
            measurements: Volume/bbox measurements (from utils.measurements)
            prompt: Optional text prompt (e.g., "Clinical indication: ...")
            max_length: Maximum report length
            return_attention: Return attention weights for visualization
        
        Returns:
            Dictionary with generated report and optional attention maps
        """
        
        if self.llm is None:
            # Mock generation for testing
            return {
                'report': "MOCK REPORT: Liver: unremarkable. Lungs: clear. No acute findings.",
                'attention_maps': None
            }
        
        # 1. Encode segmentation features to sequence
        seg_sequence = self.encode_segmentation_features(seg_features)
        
        # 2. Encode measurements
        measurement_emb = self.encode_measurements(measurements)
        
        # 3. Prepare text prompt
        if prompt is None:
            prompt = "Generate a radiology report based on the CT scan:\n"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # 4. Get LLM embeddings
        input_embeds = self.llm.get_input_embeddings()(inputs['input_ids'])
        
        # 5. Prepend segmentation and measurement embeddings
        # [seg_sequence | measurement | text_embeddings]
        combined_embeds = torch.cat([
            seg_sequence,
            measurement_emb.unsqueeze(1),
            input_embeds
        ], dim=1)
        
        # 6. Generate with cross-attention to segmentation features
        # Note: This is a simplified version. In practice, you'd need to
        # modify the LLM forward pass to insert cross-attention layers.
        # For now, we use the segmentation as prefix embeddings.
        
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode report
        report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'report': report,
            'attention_maps': None  # TODO: Extract from cross-attention layers
        }
    
    def generate_report(
        self,
        seg_output: Dict[str, Any],
        clinical_indication: Optional[str] = None,
        rag_context: Optional[str] = None
    ) -> str:
        """
        High-level interface for report generation.
        
        Args:
            seg_output: Output from SegmentationModel.forward()
            clinical_indication: Optional clinical context
            rag_context: Optional retrieved guidelines (from utils.rag)
        
        Returns:
            Generated radiology report as string
        """
        # Extract features and measurements
        seg_features = seg_output['features']['bottleneck']
        measurements = seg_output.get('measurements', {})
        
        # Build prompt
        prompt_parts = []
        if clinical_indication:
            prompt_parts.append(f"Clinical indication: {clinical_indication}")
        if rag_context:
            prompt_parts.append(f"Relevant guidelines: {rag_context}")
        prompt_parts.append("\\nGenerate radiology report:\\n")
        
        prompt = "\\n".join(prompt_parts)
        
        # Generate
        output = self.forward(
            seg_features=seg_features,
            measurements=measurements,
            prompt=prompt
        )
        
        return output['report']


class ReportGeneratorTrainer:
    """
    Training utilities for the report generator.
    """
    
    def __init__(
        self,
        model: SegmentationGuidedReportGenerator,
        learning_rate: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
    
    def compute_loss(
        self,
        seg_features: torch.Tensor,
        measurements: Dict[str, Any],
        target_report: str
    ) -> torch.Tensor:
        """
        Compute language modeling loss.
        
        Args:
            seg_features: Segmentation encoder features
            measurements: Quantitative measurements
            target_report: Ground truth report text
        
        Returns:
            loss: Cross-entropy loss
        """
        # Tokenize target
        target_tokens = self.model.tokenizer(
            target_report,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Forward pass
        # TODO: Implement teacher forcing training
        # This requires modifying forward() to accept target tokens
        
        raise NotImplementedError("Training loop to be implemented")
