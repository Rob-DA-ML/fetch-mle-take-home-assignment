import torch
from transformers import AutoTokenizer
from model import SentenceTransformerModel

def encode_sentences(sentences, model, tokenizer, device="cpu", max_length=32):
    """
    Encodes a list of sentences using the SentenceTransformerModel.
    Returns a tensor of shape [batch_size, embedding_dim].
    """
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    return embeddings

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentenceTransformerModel(model_name=model_name, pooling="mean").to(device)
    model.eval()

    # Example sentences
    sentences = [
        "Hello, how are you?",
        "I love exploring deep learning techniques.",
        "Sentence transformers encode text into embeddings."
    ]

    # Encode
    embeddings = encode_sentences(sentences, model, tokenizer, device=device)

    # Print results
    print("Embeddings shape:", embeddings.shape)
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Embedding (first 5 values): {embeddings[i][:5]}")
