## 3. Train linear probes on activations.
import torch

## Grab activations
activations = torch.load("./activations.pt")

last_token_activations = [(classification, activation[-1]) for (classification, activation) in activations]

## Get all classes of behavior
classes = set([a[0] for a in activations])
## Map classes to integers
int_to_class = {i:c for (i, c) in zip(range(0, len(classes)), classes)}
class_to_int = {c:i for (i, c) in zip(range(0, len(classes)), classes)}


## Format data as (x, y)
x = []
y = []

for (classification, activation) in last_token_activations:
    x.append(activation)
    y.append(class_to_int[classification])

## Split data
tenth = int(0.1 * len(x))

x_train = x[:8*tenth]
y_train = y[:8*tenth]

x_val = x[8*tenth:]
y_val = y[8*tenth:]


## The rest is heavily inspired by: https://github.com/soucs/linear-probe

## Standardize x-values
##  - Each feature has mean 0 and SD 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

## Now make the activations smaller
from sklearn.decomposition import PCA
pca = PCA(0.99)
x_train = pca.fit_transform(x_train)
x_val = pca.transform(x_val)

## More imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

## Probe creation & training
probe = LogisticRegression(
    solver='lbfgs',
    C=0.1,
    max_iter=1000,
    random_state=42
)

probe.fit(x_train, y_train)

y_pred = probe.predict(x_val)

print("--- Output Metrics ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0, average='macro'):.4f}")
print(f"Recall: {recall_score(y_val, y_pred, zero_division=0, average='macro'):.4f}")

## Save everything
import pickle

## Probe
with open("./linear_probe.pkl", "wb") as f:
    pickle.dump(probe, f)

## PCA
with open("./probe_pca.pkl", "wb") as f:
    pickle.dump(pca, f)

## Scalar
with open("./probe_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
