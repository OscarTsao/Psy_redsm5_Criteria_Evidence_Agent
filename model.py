import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple

class SpanBERTForDSM5Classification(nn.Module):
    def __init__(self, model_name: str = 'SpanBERT/spanbert-base-cased', num_criteria: int = 9, dropout: float = 0.1):
        super(SpanBERTForDSM5Classification, self).__init__()
        self.num_criteria = num_criteria

        self.config = AutoConfig.from_pretrained(model_name)
        self.spanbert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_criteria)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.spanbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        probs = self.sigmoid(logits)
        predictions = (probs > threshold).float()
        return predictions, probs

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

def get_model(model_name: str = 'SpanBERT/spanbert-base-cased', num_criteria: int = 9, device: str = None) -> Tuple[SpanBERTForDSM5Classification, torch.device]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = SpanBERTForDSM5Classification(model_name, num_criteria)
    model = model.to(device)

    return model, device