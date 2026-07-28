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

benchmark1 = benchmark.DeceptionBench(model_id,
                                      layers,
                                      probe,
                                      scaler,
                                      pca)


results = benchmark1.run_all(tokenized_data=_load("tokenized.pkl"))
_save(results, "results.pkl")
