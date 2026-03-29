import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertConfig, BertModel

# 1. Config and Dictionary
dna_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '[MASK]': 4}
config = BertConfig(
    vocab_size=5,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=32
)

def tokenize(seq):
    return [dna_dict[c] for c in seq]

def mask_seq(seq, mask_prob=0.15):
    tokens = tokenize(seq)
    labels = [-100] * len(tokens) # Initialize labels with ignore index

    for i in range(len(tokens)):
        if random.random() < mask_prob:
            labels[i] = tokens[i]      # The ground truth
            tokens[i] = dna_dict['[MASK]']
    
    return tokens, labels

def gen_seq(n=500, length=12):
    seqs = []
    for _ in range(n):
        s = ''.join(random.choice('ATCG') for _ in range(length))
        seqs.append(s)
    return seqs

class DNAMLM(nn.Module):
    def __init__(self, config):
        super(DNAMLM, self).__init__() # Required for nn.Module
        self.transformer = BertModel(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        outputs = self.transformer(input_ids=x)
        # We take the hidden states of all tokens
        logits = self.head(outputs.last_hidden_state) 
        return logits

def prep_batch(seq_list):
    input_ids = []
    labels = []
    for s in seq_list:
        tokens, lbls = mask_seq(s)
        input_ids.append(tokens)
        labels.append(lbls)
    return torch.tensor(input_ids), torch.tensor(labels)

def train():
    model = DNAMLM(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    sequences = gen_seq(500)

    for epoch in range(5):
        model.train()
        inputs, labels = prep_batch(sequences)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Flatten for CrossEntropy (Batch * Seq, Vocab)
        loss = loss_fn(outputs.view(-1, 5), labels.view(-1))

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "dna_pretrained.pth")
    print("Model saved as dna_pretrained.pth")

if __name__ == "__main__":
    train()
