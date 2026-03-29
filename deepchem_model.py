# train_deepchem.py

import torch
import torch.nn as nn
import deepchem as dc
import numpy as np
import random
from transformers import BertConfig, BertModel
from deepchem.models.torch_models import TorchModel

vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

def tokenize(seq):
    return [vocab[c] for c in seq]



def generate_data(n=200):
    X = []
    y = []

    for _ in range(n):
        seq = ''.join(random.choice('ATCG') for _ in range(10))
        label = 1 if seq.count('A') > 5 else 0

        X.append(tokenize(seq))
        y.append(label)

    return np.array(X), np.array(y)


class DNAClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        config = BertConfig(
            vocab_size=5,  
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32
        )

        self.transformer = BertModel(config)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        outputs = self.transformer(input_ids=x.long())
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


def load_pretrained(model, path="dna_pretrained.pth"):
    state_dict = torch.load(path)
    model.transformer.load_state_dict(state_dict, strict=False)
    print("Pretrained weights loaded!")


X, y = generate_data(200)

dataset = dc.data.NumpyDataset(X, y)



model_pt = DNAClassifier()

load_pretrained(model_pt)

def loss_fn(outputs, labels, weights):
    
    
    ce_loss = nn.CrossEntropyLoss()
    
   
    out = outputs[0]
    lab = labels[0]
    

    lab = lab.view(-1).long() 
    
    return ce_loss(out, lab)

dc_model = TorchModel(
    model=model_pt,
    loss=loss_fn,
    output_types=['prediction']
) 



dc_model.fit(dataset, nb_epoch=5)


preds = dc_model.predict(dataset)
pred_labels = np.argmax(preds, axis=1).flatten() 

accuracy = (pred_labels == y).mean()

print(f"Accuracy: {accuracy:.4f}")