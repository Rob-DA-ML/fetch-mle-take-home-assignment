import torch
import torch.nn as nn
from transformers import AutoModel

class SentenceTransformerModel(nn.Module):
    """
    A simple Sentence Transformer model that encodes input sentences into fixed-length vectors.
    """
    def __init__(self, model_name="distilbert-base-uncased", pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.transformer = AutoModel.from_pretrained(model_name)
        # Optional projection layer to reduce dimensionality
        self.projection = nn.Linear(768, 256)

    def forward(self, input_ids, attention_mask):
        # Pass tokens through the transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # Pooling strategy
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # E.g., use the CLS token representation
            pooled = last_hidden_state[:, 0]

        # Optional projection
        embeddings = self.projection(pooled)  # [batch_size, 256]
        return embeddings
