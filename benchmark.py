## 4. Run against benchmarks
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import asctime

## To silence errors
from transformers.utils import logging
logging.set_verbosity_error()

## Load trained probe
import pickle

probe = pickle.load(open("./linear_probe.pkl", "rb"))
pca = pickle.load(open("./probe_pca.pkl", "rb"))
scaler = pickle.load(open("./probe_scaler.pkl", "rb"))


## Set up model for inference
#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map={"": "cpu"},
                                             dtype=torch.float16,
                                             low_cpu_mem_usage=True)

## Register hooks to grab residual stream
temp = []

def hook(module, input, output):
    temp.append(output[0].detach())

target_layer = model.model.layers[19]
target_layer.register_forward_hook(hook)

def prompt(prompt):
    return tokenizer(prompt, return_tensors="pt")

def run(ask, num_new_tokens=128, stop_strings=None):
    return model.generate(**prompt(ask),
                          max_new_tokens=num_new_tokens,
                          stop_strings=stop_strings,
                          tokenizer=tokenizer)

## !! - Hard-coded :(
##      classification information to interpret the probe's output
int_to_class = {0: 'considered_executed', 1: 'no_consideration', 2: 'considered_dismissed'}
class_to_int = {'considered_executed': 0, 'no_consideration': 1, 'considered_dismissed': 2}

bad = [0]
good = [1, 2]

def predict_from_activation(activation):
    activation = pca.transform(scaler.transform(activation))
    return int(probe.predict(activation)[0])

## DeceptionBench first
from datasets import load_dataset

print("[+] Loading DeceptionBench...")
dataset = load_dataset("PKU-Alignment/DeceptionBench")
db_test = dataset["test"]

scores = []

for sample in db_test:
    print(f"Next sample started @ {asctime()}")
    ## Load sample
    system_prompt = sample.get("system_prompt", "You are a helpful assistant.")
    outer_prompt = sample.get("outer_prompt", "")
    format_instruction = sample.get("format", "")
    
    full_user_prompt = f"{system_prompt}\n{outer_prompt}\n\n{format_instruction}"

    ## Run sample
    output = run(full_user_prompt, num_new_tokens=1024, stop_strings=["</output>"])
    output = tokenizer.batch_decode(output)[0]

    ## ?! How do you use the linear probes now that you have them?!

    ## Remove first full context window.
    del temp[0]
        
    sample_scores = []
    ## Extract per-token probe scores
    for i in range(0, len(temp), 10):
        window = temp[i:i+10]
        score = torch.stack(window).mean(dim=0)
        score = predict_from_activation(score)
        sample_scores.append(score)
        
    scores.append([sample_scores, temp, full_user_prompt, output])
    temp = []

with open("./results.pkl", "wb") as f:
    pickle.dump(scores, f)
