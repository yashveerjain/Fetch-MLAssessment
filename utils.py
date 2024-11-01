import torch
from model import *

def init_model(vocab_size=10000, n_layers=1, d_model=512, nhead=8, embed_size=300, task="se"):
    base_model = BaseModel(vocab_size=vocab_size, n_layers=n_layers, d_model=d_model, nhead=nhead)

    if task == "se":
        task_model = SentenceEmbedding(d_model=d_model,embed_size=embed_size)
    elif task == "sc":
        task_model = SentimentClassifier(d_model=d_model, n_classes=3)
    else:
        raise ValueError("task must be 'se' or 'sc'")

    return base_model, task_model
