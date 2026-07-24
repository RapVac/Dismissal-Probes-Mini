## 1. Get labeled list of deception modes (accepted, rejected, n/a) + associated CoT.
import json

print("[+] Reading training data...")
## Shamelessly taken from Sam
train_str = open("./dismissal_triplets.jsonl", "r").read().strip().split("\n")
train = []

for json_str in train_str:
    train.append(json.loads(json_str))

## extract string of conversation from list of dicts
def str_from_json_convo(convo: list):
    output = ""
    for turn in convo:
        output += turn["role"].upper() + ": " + turn["content"] + "\n"
    return output

## extract (label, convo_str) from original dataset
def get_labeled_sets(json_list):
    data = []
    for j in json_list:
        tmp = []
        tmp.append(j["gold_outcome"])
        tmp.append(str_from_json_convo(j["messages"]))
        data.append(tmp)
    return data

training_convos = get_labeled_sets(train)
print("[+] Training data ready.")


## 2. Extract activations from pass thru target OS model.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

## Configure tokenizer for batching
tokenizer = AutoTokenizer.from_pretrained(model_id)

## Initiate model
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map={"": "cpu"},
                                             dtype=torch.float16,
                                             low_cpu_mem_usage=True)

temp = []

def hook(module, input, output):
    temp.append(output.detach())

## !! - grab residual stream
target_layer = model.model.layers[19]
target_layer.register_forward_hook(hook)

def prompt(prompt):
    return tokenizer(prompt, return_tensors="pt")

## Run across deception examples
get_activations = True
activations = []

if get_activations:
    print("[+] Extracting activations:")
    with torch.no_grad():
        for i, (classification, convo) in tqdm(enumerate(training_convos), total=len(training_convos)):
            ## Run thru model
            model(**prompt(convo))
            activations.append((classification, temp[-1].squeeze(0)))

        torch.save(activations, "activations.pt")
