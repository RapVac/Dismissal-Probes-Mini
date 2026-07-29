import pickle
import json

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

probe = _load("probe.pkl")
scaler = _load("scaler.pkl")
pca = _load("pca.pkl")

##benchmark1 = benchmark.DeceptionBench(model_id,
##                                      layers,
##                                      probe,
##                                      scaler,
##                                      pca)


#benchmark1.prepare_dataset()
#results = benchmark1.run_all(24)
#_save(results, "results.pkl")


##====================================

results = _load("results.pkl")

results = [("deception_bench", result) for result in results]
model, tokenizer = run_activations.init_model(model_id)
hooks = run_activations.add_hooks(model, *layers)

results = run_activations.get_labeled_activations(model, tokenizer, results, len(layers))
results = run_activations.get_activation_by_index(results, 0)
_save(labeled_activations, "benchmark_activations.pkl")
