# Multi-Task Sentence Transformer

This repository demonstrates:
1. **Step 1**: A basic Sentence Transformer that encodes sentences into fixed-length embeddings.
2. **Step 2**: A multi-task expansion (e.g., sentence classification + token-level classification).
3. **Step 3**: Additional discussion (fine-tuning strategies, deciding between multi-task vs. single-task, and handling data imbalance). See [DISCUSSION.md](DISCUSSION.md).

## Getting Started

1. **Clone** the repository:
   ```bash
   git clone <your-repo-url>
   cd multi-task-sentence-transformer
2. pip install -r requirements.txt
3. Explore the notebooks:
   notebooks/01_sentence_transformer_demo.ipynb
   notebooks/02_multitask_demo.ipynb
4. Run inference scripts
   python src/inference.py
   python src/inference_multitask.py
