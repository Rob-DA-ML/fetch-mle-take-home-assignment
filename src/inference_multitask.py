import torch
from transformers import AutoTokenizer
from multitask_model import MultiTaskSentenceTransformer

def run_multitask_inference(sentences, model, tokenizer, device="cpu", max_length=32):
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
        outputs = model(input_ids, attention_mask)

    task_a_logits = outputs["task_a_logits"]  # [batch_size, num_classes]
    task_b_logits = outputs["task_b_logits"]  # [batch_size, seq_len, num_labels]
    return task_a_logits, task_b_logits, encoded

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "distilbert-base-uncased"

    # Example: 3 classes for Task A, 5 labels for Task B
    model = MultiTaskSentenceTransformer(
        model_name=model_name,
        num_classes_task_a=3,
        num_labels_task_b=5,
        pooling="mean"
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sentences = [
        "Barack Obama was the 44th President of the United States.",
        "I love exploring advanced transformer architectures."
    ]
    task_a_logits, task_b_logits, inputs = run_multitask_inference(
        sentences, model, tokenizer, device=device
    )

    print("=== Task A (Sentence Classification) Logits ===")
    print(task_a_logits)
    print("\n=== Task B (Token-Level) Logits Shape ===")
    print(task_b_logits.shape)

    tokens_batch = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]
    for i, tokens in enumerate(tokens_batch):
        print(f"\nSentence {i+1} tokens:", tokens)
        print("NER logits per token:", task_b_logits[i].shape)
