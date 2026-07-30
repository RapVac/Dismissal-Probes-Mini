import torch

from scripts import run_activations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def labeled_dataset_to_x_y_pairs(labeled_dataset, class_a, class_b, default_val):
    x = []
    y = []
    classes = {class_a: 0, class_b: 1}
    for (classification, activation) in labeled_dataset:
        x.append(activation.reshape(-1))
        y.append(classes.get(classification, default_val))

    return x, y

def fit_probe(x, y, test_size=0.2, random_state=42, probe_iter=1000, get_probe_stats=True):
    x_train, x_val, y_train, y_val = train_test_split(x,
                                                      y,
                                                      test_size=test_size,
                                                      random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    pca = PCA(0.99)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)

    probe = LogisticRegression(solver='lbfgs',
                               C=0.1,
                               max_iter=probe_iter,
                               random_state=random_state)
    probe.fit(x_train, y_train)

    output = [scaler, pca, probe]

    if get_probe_stats:
        y_pred = probe.predict(x_val)
        y_pred_proba = probe.predict_proba(x_val)[:, 1]
        stats = {}
        stats["accuracy"] = accuracy_score(y_val, y_pred)
        stats["precision"] = precision_score(y_val, y_pred, zero_division=0, average='macro')
        stats["recall"] = recall_score(y_val, y_pred, zero_division=0, average='macro')
        stats["roc-auc"] = roc_auc_score(y_val, y_pred_proba)
        output.append(stats)

    return output

def train_probe(selected_layer, layers, labeled_deception_examples, model, tokenizer):
    ## (class, activations[])
    labeled_activations = run_activations.get_labeled_activations(model,
                                                                  tokenizer,
                                                                  labeled_deception_examples,
                                                                  len(layers))
    ## (class, activations[selected_layer])
    labeled_activations = run_activations.get_activation_by_index(labeled_activations, selected_layer)

    ## 3. Train linear probes.
    x1, y1 = train_probes.labeled_dataset_to_x_y_pairs(labeled_activations, "considered_executed", "no_consideration", 0)

    scaler, pca, probe, stats = train_probes.fit_probe(x1, y1)
    print_stats(stats)
    return scaler, pca, probe
