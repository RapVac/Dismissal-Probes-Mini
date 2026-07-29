import torch

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer

def syntactic_chunks(output, activations, model_id, response_start_string):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    response_start_token = tokenizer(response_start_string)["input_ids"]
    tokenized_output = tokenizer(output)["input_ids"]
    punctuation = []
    for char in ".,;":
        punctuation.append(tokenizer(char)["input_ids"][0])
    
    response_start = -1
    for i in range(0, len(tokenized_output), len(response_start_token)):
        window = tokenized_output[i: i + len(response_start_token)]
        if window == response_start_token:
            response_start = i

    response = tokenized_output[response_start:]
    response_activations = activations[response_start:]
    i = 0
    anchor = 0
    syntactic_chunks = []
    while i < len(response):
        if response[i] in punctuation:
            syntactic_chunks.append(response_activations[anchor:i].mean(dim=0))
            anchor = i
        i += 1

    return syntactic_chunks
