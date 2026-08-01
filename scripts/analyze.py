from scripts import run_activations
import tqdm

def get_syntactic_chunks(output, activations, tokenizer, response_start_string="<think>"):
    response_start_token = tokenizer(response_start_string)["input_ids"]
    tokenized_output = tokenizer(output)["input_ids"]
    punctuation = []
    for char in ".,;":
        punctuation.append(tokenizer(char)["input_ids"][0])

    ## Slide a window across response to find last instance of `start_string'
    response_start = -1
    for i in range(0, len(tokenized_output), len(response_start_token)):
        window = tokenized_output[i: i + len(response_start_token)]
        if window == response_start_token:
            response_start = i

    ## Seperate & average activations based on punctuation.
    response = tokenized_output[response_start:]
    response_activations = activations[response_start:]
    i = 0
    anchor = 0
    syntactic_chunks = []
    while i < len(response):
        if response[i] in punctuation:
            syntactic_chunks.append(response_activations[anchor:i].mean(dim=0))
            anchor = i
        i += 1

    return syntactic_chunks

def benchmark_output_to_probe_scores(benchmark_output, layers, model_id, probes):
    ## a. Get activations.
    ## Second item will be replaced with activations.
    benchmark_output = [(r[1], r[1]) for r in benchmark_output]
    model, tokenizer = run_activations.init_model(model_id)
    hooks = run_activations.add_complete_hooks(model, *layers)

    benchmark_activations = run_activations.get_labeled_activations(model, tokenizer, benchmark_output, len(layers))
    results = []    

    for i, layer in enumerate(layers):
        layer_results = run_activations.get_activation_by_index(benchmark_activations, i)
        ## Results -> (output, raw_activations)

        ## b. Syntactic chunking.
        chunked = []
        for (output, activations) in tqdm.tqdm(layer_results, desc="Breaking activations into syntactic chunks: "):
            chunks = get_syntactic_chunks(output, activations.squeeze(0), tokenizer)
            chunked.append((output, chunks))

        scored = []
        scaler, pca, probe = probes[layer]

        for (output, aggregates) in tqdm.tqdm(chunked, desc="Scoring chunks: "):
            chunk_scores = []
            for aggregate in aggregates:
                scaler_data = scaler.transform(aggregate.unsqueeze(0))
                pca_data = pca.transform(scaler_data)
                prediction = int(probe.predict(pca_data)[0])
                
                chunk_scores.append(prediction)

            scored.append((output, chunk_scores))

        results.append(scored)

    return results

