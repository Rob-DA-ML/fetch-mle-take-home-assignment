import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskSentenceTransformer(nn.Module):
    """
    Multi-task model with a shared transformer backbone:
      - Task A: Sentence-level classification
      - Task B: Token-level classification (e.g., NER)
    """
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_classes_task_a=3,
        num_labels_task_b=5,
        pooling="mean"
    ):
        super().__init__()
        self.pooling = pooling
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Optional projection layer for sentence embeddings
        self.projection = nn.Linear(768, 256)
        
        # Task A: Sentence-level classification head
        self.classifier_task_a = nn.Linear(256, num_classes_task_a)
        
        # Task B: Token-level classification head (e.g., NER)
        self.classifier_task_b = nn.Linear(768, num_labels_task_b)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # --- Task A ---
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_embedding = sum_embeddings / sum_mask
        else:
            pooled_embedding = last_hidden_state[:, 0]
        
        # Projection for sentence-level tasks
        pooled_embedding = self.projection(pooled_embedding)  # [batch_size, 256]
        task_a_logits = self.classifier_task_a(pooled_embedding)

        # --- Task B ---
        # Token-level classification (e.g., NER) uses the unpooled hidden states
        task_b_logits = self.classifier_task_b(last_hidden_state)  # [batch_size, seq_len, num_labels]

        return {
            "task_a_logits": task_a_logits,
            "task_b_logits": task_b_logits
        }
