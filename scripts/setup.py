import json

def json_to_str(convo: list):
    output = ""
    for turn in convo:
        output += turn["role"].upper() + ": " + turn["content"] + "\n"
    return output

def dataset_to_labeled_pairs(dataset: list):
    labeled_pairs = []
    for item in dataset:
        labeled_pairs.append((item["gold_outcome"],
                             json_to_str(item["messages"])))

    return labeled_pairs

def load_dataset(dataset_path):
    dataset_str = open(dataset_path, "r").read().strip().split("\n")
    dataset = []
    for json_str in dataset_str:
        dataset.append(json.loads(json_str))

    dataset = dataset_to_labeled_pairs(dataset)
    return dataset
