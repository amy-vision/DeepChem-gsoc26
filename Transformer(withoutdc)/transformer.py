from transformers import BertConfig,BertModel
import torch
import torch.nn as nn
import random
import torch 
import torch.optim as optim

#building the tokenizer
dict={'A':0,'T':1,'G':2,'C':3}
def tokenize(seq):
    return [dict[s] for s in seq]
   
    
seq="ATGCAA"
print(tokenize(seq))



config=BertConfig(
    vocab_size=4, #no of possible tokens
    hidden_size=64, # size of internal vector 
    num_hidden_layers=2, #no of transformer layers
    num_attention_heads=2 #no of heads
)
model=BertModel(config)

tokens=torch.tensor([tokenize(seq)])
output=model(input_ids=tokens)
# print(output.last_hidden_state.shape)

class DNAClassifier(nn.Module):
    def __init__(self,transformer):
        super().__init__()
        self.transformer=transformer #bert transformer 
        self.classifier=nn.Linear(64,2) # takes 64 numbers and give output as 2 numbers

    def forward(self,x):
            outputs=self.transformer(input_ids=x)
            cls_token=outputs.last_hidden_state[:,0,:] #takes the first letter of all seq
            return self.classifier(cls_token)
        

def generate_data(n=100):
    X=[]
    y=[]
    for _ in range(n):
         seq=''.join(random.choice('ATCG') for _ in range(10))
         label=1 if seq.count('A')>5 else 0
         X.append(tokenize(seq))
         y.append(label)
         return X,y
    

X,y=generate_data(200)
X=torch.tensor(X)
y=torch.tensor(y)

model=DNAClassifier(model)
optimizer=optim.Adam(model.parameters(),lr=0.001) #Adam algorithm to manage model's weight to reduce the error 
loss_fn=nn.CrossEntropyLoss()#for calculating loss

for epoch in range(5):
     optimizer.zero_grad() #clears out memory of prev losses
     preds=model(X)#looks at model and give its current best guess
     loss=loss_fn(preds,y)#checks the loss
     loss.backward()#goes backward to check which neuron caused the error
     optimizer.step()#changes weight to make it more accurate
     print("Loss:",loss.item())

        
    
