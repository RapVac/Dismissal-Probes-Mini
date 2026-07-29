import pickle
import json
import tqdm

from scripts import setup, run_activations, train_probes, benchmark, analyze

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

##probe = _load("probe.pkl")
##scaler = _load("scaler.pkl")
##pca = _load("pca.pkl")

benchmark1 = benchmark.DeceptionBench(model_id)

benchmark1.prepare_dataset()
results = benchmark1.run_all()
save(results, "results.pkl")


##====================================

##results = _load("results.pkl")
##
##results = [(r[1], r[1]) for r in results]
##
##model, tokenizer = run_activations.init_model(model_id)
##hooks = run_activations.add_complete_hooks(model, *layers)
##
##results = run_activations.get_labeled_activations(model, tokenizer, results, len(layers))
##results = run_activations.get_activation_by_index(results, 0)
##_save(results, "benchmark_activations.pkl")

##====================================

##labeled_activations = _load("activations.pkl")
##
##x1, y1 = train_probes.labeled_dataset_to_x_y_pairs(labeled_activations, "considered_executed", "no_consideration", 1)
##
##scaler, pca, probe, stats = train_probes.fit_probe(x1, y1)
##print_stats(stats)
##
##_save(probe, "probe.pkl")
##_save(scaler, "scaler.pkl")
##_save(pca, "pca.pkl")
##
##benchmark_activations = _load("benchmark_activations.pkl")
##
##chunked = []
##
##for (output, activations) in tqdm.tqdm(benchmark_activations):
##    tmp = analyze.syntactic_chunks(output, activations.squeeze(0), model_id, "<think>")
##    chunked.append((output, tmp))
##
##scored = []
##
##for (output, aggregates) in tqdm.tqdm(chunked):
##    tmp = []
##    for aggregate in aggregates:
##        tmp.append(int(probe.predict(pca.transform(scaler.transform(aggregate.unsqueeze(0))))[0]))
##
##    scored.append((output, tmp))
