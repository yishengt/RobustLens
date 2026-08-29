"""
AI Image Detector - SigLIP2 + DINOv2 Ensemble with LoRA

This model detects AI-generated images using an ensemble of:
- SigLIP2-SO400M (semantic features)
- DINOv2-Large (self-supervised visual features)

Both backbones use LoRA adapters for efficient fine-tuning.
"""

import torch
import torch.nn as nn
import math
from torch.amp import autocast

import timm
from transformers import AutoProcessor, SiglipVisionModel
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from PIL import Image


class LoRALinear(nn.Module):
    """Custom LoRA implementation for DINOv2 QKV layers."""
    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        
        for p in self.original.parameters():
            p.requires_grad = False
        
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class ClassificationHead(nn.Module):
    """MLP classification head with LayerNorm and dropout."""
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        return self.head(x).squeeze(-1)


class EnsembleAIDetector(nn.Module):
    """Ensemble model combining SigLIP2 and DINOv2 for AI image detection."""
    
    def __init__(self, siglip_model_name: str, dinov2_model_name: str, image_size: int = 392):
        super().__init__()
        
        # SigLIP2 backbone
        self.siglip = SiglipVisionModel.from_pretrained(
            siglip_model_name, 
            torch_dtype=torch.bfloat16
        )
        self.siglip_dim = self.siglip.config.hidden_size
        
        # DINOv2 backbone
        self.dinov2 = timm.create_model(
            dinov2_model_name, 
            pretrained=True, 
            num_classes=0,
            img_size=image_size
        )
        self.dinov2_dim = self.dinov2.num_features
        
        # Classification head
        self.classifier = ClassificationHead(self.siglip_dim + self.dinov2_dim)
    
    def forward(self, siglip_pixels, dinov2_pixels):
        # Extract features
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        
        # Combine and classify
        combined = torch.cat([siglip_features.float(), dinov2_features], dim=-1)
        logits = self.classifier(combined)
        
        return logits, siglip_features, dinov2_features


def create_model_with_lora(
    siglip_model_name: str = "google/siglip2-so400m-patch14-384",
    dinov2_model_name: str = "vit_large_patch14_dinov2.lvd142m",
    image_size: int = 392,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1
) -> EnsembleAIDetector:
    """Create the model with LoRA adapters applied."""
    
    model = EnsembleAIDetector(siglip_model_name, dinov2_model_name, image_size)
    
    # Apply LoRA to SigLIP using PEFT
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none"
    )
    model.siglip = get_peft_model(model.siglip, lora_config)
    
    # Apply LoRA to DINOv2 (custom implementation for QKV layers)
    for name, module in model.dinov2.named_modules():
        if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, lora_rank, lora_alpha, lora_dropout)
    
    return model


def create_transforms(image_size: int = 392):
    """Create preprocessing transforms for DINOv2."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AIImageDetector:
    """High-level API for AI image detection."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to pytorch_model.pt
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Create model
        self.model = create_model_with_lora(
            siglip_model_name=config.get('siglip_model', 'google/siglip2-so400m-patch14-384'),
            dinov2_model_name=config.get('dinov2_model', 'vit_large_patch14_dinov2.lvd142m'),
            image_size=config.get('image_size', 392),
            lora_rank=config.get('lora_rank', 32),
            lora_alpha=config.get('lora_alpha', 64),
            lora_dropout=config.get('lora_dropout', 0.1),
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create processors
        self.siglip_processor = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384')
        self.dinov2_transform = create_transforms(config.get('image_size', 392))
        
        print(f"Model loaded on {self.device}")
    
    @torch.no_grad()
    def predict(self, image) -> dict:
        """
        Predict whether an image is AI-generated.
        
        Args:
            image: PIL Image, path to image, or URL
            
        Returns:
            dict with keys:
                - probability: float, P(AI-generated)
                - prediction: str, "ai-generated" or "real"
                - confidence: float, confidence score
        """
        # Load image if needed
        if isinstance(image, str):
            if image.startswith('http'):
                import requests
                from io import BytesIO
                response = requests.get(image)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        siglip_inputs = self.siglip_processor(images=image, return_tensors="pt")
        siglip_pixels = siglip_inputs["pixel_values"].to(self.device)
        dinov2_pixels = self.dinov2_transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with autocast('cuda', enabled=self.device.type == 'cuda'):
            logits, _, _ = self.model(siglip_pixels, dinov2_pixels)
        
        probability = torch.sigmoid(logits).item()
        prediction = "ai-generated" if probability > 0.5 else "real"
        confidence = probability if probability > 0.5 else 1 - probability
        
        return {
            "probability": probability,
            "prediction": prediction,
            "confidence": confidence
        }
    
    def __call__(self, image):
        """Shorthand for predict()."""
        return self.predict(image)