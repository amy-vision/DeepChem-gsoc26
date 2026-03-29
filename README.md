DNA Foundation Model with DeepChem
This project implements a DNA foundation model using PyTorch and HuggingFace Transformers, with integration into DeepChem for downstream training and evaluation.
The model is pretrained using masked language modeling (MLM) on DNA sequences and then fine-tuned for supervised tasks such as classification. The objective is to build a reusable and extensible biological sequence model within the DeepChem ecosystem.

This project follows a standard foundation model pipeline:
DNA Sequence → Tokenizer → Pretraining (MLM) → Transformer → Fine-tuning → DeepChem → Evaluation
The approach separates self-supervised pretraining from supervised fine-tuning, enabling the model to learn general sequence representations before being adapted to specific tasks.
