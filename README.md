Everything should be well-commented.

1. ``setup.py`` extracts training data and runs it through Llama to get activations.
2. ``train_probes.py`` trains a linear probe on that data.
3. ``benchmark.py`` runs these probes against some other deception benchmarks

``run.sh`` runs all three in order.

These scripts also save their results in several files, of which there are examples included.

``activations.pt`` contains some saved activations from my runs.
``linear_probe.pkl`` contains the saved probe.

``probe_scaler.pkl`` contains additional utilities for the probe (it's role is evident in the code).
``probe_pca.pkl`` same as above.

``results.pkl`` contains the probe scores for a run across the test set from DeceptionBench.
