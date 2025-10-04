import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Tuple, Optional


class BERTForPairwiseClassification(nn.Module):
    """Binary classifier for (post, criterion) pairs with a single logit output."""

    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, threshold: float = 0.5):
        logits = self.forward(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        return preds, probs

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_with_logits(inputs, targets)

        pt = torch.exp(-bce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss that scales the focusing parameter per example."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_with_logits(inputs, targets)
        probs = torch.sigmoid(inputs).clamp(min=1e-6, max=1 - 1e-6)
        pt = targets * probs + (1 - targets) * (1 - probs)

        adaptive_gamma = self.gamma + self.delta * (1 - pt)
        modulating_factor = (1 - pt) ** adaptive_gamma
        loss = self.alpha * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss for handling class imbalance."""

    def __init__(self, pos_weight: float = 10.0, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(self.pos_weight, device=inputs.device, dtype=inputs.dtype)
        loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight, reduction=self.reduction
        )
        return loss


class HybridBCEFocalLoss(nn.Module):
    """Hybrid loss combining BCE and Focal Loss with configurable weights."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_component = self.bce_loss(inputs, targets)
        focal_component = self.focal_loss(inputs, targets)
        return self.bce_weight * bce_component + (1 - self.bce_weight) * focal_component


class HybridBCEAdaptiveFocalLoss(nn.Module):
    """Hybrid loss combining BCE and Adaptive Focal Loss with configurable weights."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.adaptive_focal_loss = AdaptiveFocalLoss(alpha=alpha, gamma=gamma, delta=delta, reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_component = self.bce_loss(inputs, targets)
        focal_component = self.adaptive_focal_loss(inputs, targets)
        return self.bce_weight * bce_component + (1 - self.bce_weight) * focal_component

    def update_weights(
        self,
        *,
        bce_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ):
        if bce_weight is not None:
            self.bce_weight = float(bce_weight)
        if alpha is not None:
            self.adaptive_focal_loss.alpha = float(alpha)
        if gamma is not None:
            self.adaptive_focal_loss.gamma = float(gamma)
        if delta is not None:
            self.adaptive_focal_loss.delta = float(delta)


class DynamicLossFactory:
    """Factory for creating loss functions with hyperparameter optimization support."""

    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """Create a loss function based on type and parameters."""
        loss_type = loss_type.lower()

        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss(reduction=kwargs.get('reduction', 'mean'))

        elif loss_type == 'weighted_bce':
            return WeightedBCELoss(
                pos_weight=kwargs.get('pos_weight', 10.0),
                reduction=kwargs.get('reduction', 'mean')
            )

        elif loss_type == 'focal':
            return FocalLoss(
                alpha=kwargs.get('alpha', 0.25),
                gamma=kwargs.get('gamma', 2.0),
                reduction=kwargs.get('reduction', 'mean')
            )

        elif loss_type == 'adaptive_focal':
            return AdaptiveFocalLoss(
                alpha=kwargs.get('alpha', 0.25),
                gamma=kwargs.get('gamma', 2.0),
                delta=kwargs.get('delta', 1.0),
                reduction=kwargs.get('reduction', 'mean')
            )

        elif loss_type == 'hybrid_bce_focal':
            return HybridBCEFocalLoss(
                bce_weight=kwargs.get('bce_weight', 0.5),
                alpha=kwargs.get('alpha', 0.25),
                gamma=kwargs.get('gamma', 2.0),
                reduction=kwargs.get('reduction', 'mean')
            )

        elif loss_type == 'hybrid_bce_adaptive_focal':
            return HybridBCEAdaptiveFocalLoss(
                bce_weight=kwargs.get('bce_weight', 0.5),
                alpha=kwargs.get('alpha', 0.25),
                gamma=kwargs.get('gamma', 2.0),
                delta=kwargs.get('delta', 1.0),
                reduction=kwargs.get('reduction', 'mean')
            )

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    @staticmethod
    def get_loss_param_ranges(loss_type: str) -> dict:
        """Get parameter ranges for hyperparameter optimization."""
        loss_type = loss_type.lower()

        base_ranges = {
            'bce': {},

            'weighted_bce': {
                'pos_weight': {'method': 'uniform', 'low': 1.0, 'high': 20.0}
            },

            'focal': {
                'alpha': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'gamma': {'method': 'uniform', 'low': 0.5, 'high': 5.0}
            },

            'adaptive_focal': {
                'alpha': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'gamma': {'method': 'uniform', 'low': 0.5, 'high': 5.0},
                'delta': {'method': 'uniform', 'low': 0.1, 'high': 3.0}
            },

            'hybrid_bce_focal': {
                'bce_weight': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'alpha': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'gamma': {'method': 'uniform', 'low': 0.5, 'high': 5.0}
            },

            'hybrid_bce_adaptive_focal': {
                'bce_weight': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'alpha': {'method': 'uniform', 'low': 0.1, 'high': 0.9},
                'gamma': {'method': 'uniform', 'low': 0.5, 'high': 5.0},
                'delta': {'method': 'uniform', 'low': 0.1, 'high': 3.0}
            }
        }

        return base_ranges.get(loss_type, {})


def get_pairwise_model(
    model_name: str = 'bert-base-uncased',
    device: Optional[str] = None,
    dropout: float = 0.1,
) -> Tuple[BERTForPairwiseClassification, torch.device]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = BERTForPairwiseClassification(model_name, dropout=dropout)
    model = model.to(device)

    # Hardware optimizations
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Check for bfloat16 support (Ampere+ GPUs)
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            print(f"BFloat16 support detected on {torch.cuda.get_device_name()}")
        else:
            print(f"BFloat16 not supported on {torch.cuda.get_device_name()}, using float16")

    return model, device


def optimize_hardware_settings():
    """Apply hardware optimizations for maximum training efficiency."""
    if torch.cuda.is_available():
        # Memory management optimizations
        torch.cuda.empty_cache()

        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True

        # Set optimal number of threads
        if hasattr(torch, 'set_num_threads'):
            cpu_count = os.cpu_count() or 1
            torch.set_num_threads(max(1, min(8, cpu_count)))

        print(f"Hardware optimizations applied for {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available - running on CPU")
