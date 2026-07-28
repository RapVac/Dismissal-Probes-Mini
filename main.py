import json
import pickle

from scripts import setup, run_activations, train_probes, benchmark

## 0. Load config file.
configuration = json.loads(open("./config.json", "r").read().strip())
globals().update(configuration)

def _save(l, name):
    with open(project_home + stored_data + name, "wb") as f:
        pickle.dump(l, f)

def _load(name):
    with open(project_home + stored_data + name, "rb") as f:
        file = pickle.load(f)
    return file

def print_stats(stats):
    for key in stats.keys():
        print(f"{key}: {round(stats[key], 4)}")

## 1. Get labeled list of deception modes (accepted, rejected, n/a) + associated CoT.
activation_dataset = setup.load_dataset(project_home + training_data + activation_dataset)

## 2. Extract activations from pass thru target OW model.
model, tokenizer = run_activations.init_model(model_id)
hooks = run_activations.add_hooks(model, *layers)

labeled_activations = run_activations.get_labeled_activations(model, tokenizer, activation_dataset, len(layers))
labeled_activations = run_activations.get_activation_by_index(labeled_activations, 0)
_save(labeled_activations, "activations.pkl")

## 3. Train linear probes.
x1, y1 = train_probes.labeled_dataset_to_x_y_pairs(labeled_activations, "considered_executed", "no_consideration", 0)

scaler, pca, probe, stats = train_probes.fit_probe(x1, y1)
print_stats(stats)

_save(probe, "probe.pkl")
_save(scaler, "scaler.pkl")
_save(pca, "pca.pkl")

## 4. Run Against benchmarks.
benchmark1 = benchmark.DeceptionBench(model_id,
                                      layers,
                                      probe,
                                      scaler,
                                      pca)

results = benchmark1.run_all()
_save(results, "results.pkl")
