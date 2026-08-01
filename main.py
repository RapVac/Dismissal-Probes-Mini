import json
import pickle
import time

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

def main():
    ## 1. Get labeled list of deception modes (accepted, rejected, n/a) + associated CoT.
    ## (class, prompt)
    dataset = setup.load_dataset(project_home + training_data + activation_dataset)

    ## 2. Extract activations from pass thru target OW model.
    model, tokenizer = run_activations.init_model(model_id)
    hooks = run_activations.add_hooks(model, *layers)

    probes = {}

    layer_probes = train_probes.train_probe(layers, dataset, model, tokenizer)

    for ((scaler, pca, probe), layer) in zip(layer_probes, layers):
        probes[layer] = (scaler, pca, probe)

        _save(probe, f"probe_l{layer}.pkl")
        _save(scaler, f"scaler_l{layer}.pkl")
        _save(pca, f"pca_l{layer}.pkl")

    ## 4. Run Against benchmarks.
    print(f"[+] Started benchmark...")
    benchmark1 = benchmark.DeceptionBench(model_id)

    benchmark1.prepare_dataset()
    deception_bench_output = benchmark1.run_all()
    _save(deception_bench_output, "deception_bench_output.pkl")
    benchmark1.kill_vllm()
    
    print(f"[+] Started analysis...")
    for i, layer in enumerate(layers):
        print(f"[+] {time.asctime()} | Now on layer {layer}.")
        results = analyze.benchmark_output_to_probe_scores(deception_bench_output,
                                                           i,
                                                           layers,
                                                           model_id,
                                                           probes[layer][0],
                                                           probes[layer][1],
                                                           probes[layer][2])
        _save(results, f"results_l{layer}.pkl")

if __name__ == "__main__":
    main()
