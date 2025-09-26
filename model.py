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


class HybridLoss(nn.Module):
    """Blend BCEWithLogitsLoss and AdaptiveFocalLoss with configurable weights."""

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
    return model, device