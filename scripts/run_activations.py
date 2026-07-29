import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def init_model(model_id, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map=device_map,
                                                 dtype=torch.float16,
                                                 attn_implementation="sdpa")
    return model, tokenizer

activations = []

def hook(module, input, output):
    ## Grabs last token activations fromr residual stream
    activations.append(output[:, -1, :].detach().to('cpu'))

def complete_hook(module, input, output):
    activations.append(output.detach().to('cpu'))

def add_hooks(model, *layers):
    hooks = []
    for layer in layers:
        hooks.append(model.model.layers[layer].register_forward_hook(hook))

    return hooks

def add_complete_hooks(model, *layers):
    hooks = []
    for layer in layers:
        hooks.append(model.model.layers[layer].register_forward_hook(complete_hook))

    return hooks

def get_prompt(prompt, tokenizer):
    return tokenizer(prompt, return_tensors="pt")

def get_labeled_activations(model, tokenizer, dataset, num_layers):
    labeled_activations = []
    with torch.no_grad():
        for i, (classification, convo) in tqdm(enumerate(dataset), total=len(dataset)):
            model(**get_prompt(convo, tokenizer))
            labeled_activations.append((classification, activations[-num_layers:]))

    return labeled_activations

## With multiple layers, the labels will have multiple activations.
## This helper extracts the last-token activation for one layer only.
def get_activation_by_index(labeled_activations, index):
    new_activations = []
    for (classification, activations) in labeled_activations:
        new_activations.append((classification, activations[index]))

    return new_activations
