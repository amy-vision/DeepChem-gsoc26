#testing hugging face 
from transformers import AutoTokenizer,AutoModel
import torch
#taking the pretrained model and tokenizer
Model=AutoModel.from_pretrained("bert-base-uncased")
Tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
#input is hello world token size=4 adds cls(start) and sep(end)
# return type is tensor instead of list of integers
inputs=Tokenizer("hello world",return_tensors="pt")
#unpack the dictionary of tensors and passes to the model 
outputs=Model(**inputs)
#output of final bert layer 
print(outputs.last_hidden_state.shape)