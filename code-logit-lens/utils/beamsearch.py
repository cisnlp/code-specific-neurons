"""
Beam Search

Description:
This code performs beam search for the next sequence of tokens given the intermediate layer_id.

Copyright:
Chengzhi Zhong (https://github.com/Sayn-Wittgenstein)

License: No explicit license provided.

Change Log:
- V 1.0: https://github.com/Sayn-Wittgenstein/latent_language_of_multilingual_model/blob/main/Translation.ipynb
- V 1.1: here.
"""


import torch
import torch.nn.functional as F

def custom_beam_search(model, tokenizer, device, text, layer_id, num_beams=3, max_length=2):
    
    # Convert text to tokens
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # Initial single token candidates and probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        layer_output = model.model.layers[layer_id].output
        normed = model.model.norm(layer_output)

        # Logits = [1, tokens, vocab]
        logits = model.lm_head(normed)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, num_beams, dim=1)

    # Initialize candidates and probabilities directly
    candidates = [[top_indices[0, i].item()] for i in range(num_beams)]
    probabilities = [top_probs[0, i].item() for i in range(num_beams)]

    # Iteratively update candidates
    for step in range(1, max_length):
        new_candidates = []
        new_probabilities = []

        for i in range(len(candidates)):
            candidate_ids = torch.cat([input_ids.squeeze(0), torch.tensor(candidates[i], device=device)], dim=0).unsqueeze(0)
            with torch.no_grad():
                outputs = model(candidate_ids)
                layer_output = model.model.layers[layer_id].output
                normed = model.model.norm(layer_output)

                logits = model.lm_head(normed)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                top_probs, top_indices = torch.topk(probs, num_beams, dim=1)

            for j in range(num_beams):
                new_candidates.append(candidates[i] + [top_indices[0, j].item()])
                new_probabilities.append(probabilities[i] * top_probs[0, j].item())

        candidates = new_candidates
        probabilities = new_probabilities

        # Keep only the best candidates
        top_indices = torch.topk(torch.tensor(probabilities), num_beams, largest=True).indices
        candidates = [candidates[i] for i in top_indices]
        probabilities = [probabilities[i] for i in top_indices]

    next_texts = [tokenizer.decode(candidate) for candidate in candidates]
    return list(zip(next_texts, candidates, probabilities))



def batch_custom_beam_search(model, tokenizer, device, texts, layer_id, num_beams=3, max_length=2):

    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the input texts as a batch
    input_ids = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)

    # Initial single token candidates and probabilities
    with torch.no_grad():
        outputs = model(input_ids)
        layer_output = model.model.layers[layer_id].output
        normed = model.model.norm(layer_output)

        # Logits = [batch_size, tokens, vocab]
        logits = model.lm_head(normed)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        top_probs, top_indices = torch.topk(probs, num_beams, dim=1)

    # Initialize candidates and probabilities for each text in the batch
    batch_size = input_ids.size(0)
    candidates = [[ [top_indices[b, i].item()] for i in range(num_beams) ] for b in range(batch_size)]
    probabilities = [[ top_probs[b, i].item() for i in range(num_beams) ] for b in range(batch_size)]

    # Iteratively update candidates for each text in the batch
    for step in range(1, max_length):
        new_candidates = [[] for _ in range(batch_size)]
        new_probabilities = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            for i in range(len(candidates[b])):
                candidate_ids = torch.cat([
                    input_ids[b].unsqueeze(0),
                    torch.tensor(candidates[b][i], device=device).unsqueeze(0)
                ], dim=1)

                with torch.no_grad():
                    outputs = model(candidate_ids)
                    layer_output = model.model.layers[layer_id].output
                    normed = model.model.norm(layer_output)

                    logits = model.lm_head(normed)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    top_probs, top_indices = torch.topk(probs, num_beams, dim=1)

                for j in range(num_beams):
                    new_candidates[b].append(candidates[b][i] + [top_indices[0, j].item()])
                    new_probabilities[b].append(probabilities[b][i] * top_probs[0, j].item())

        for b in range(batch_size):
            # Keep only the best candidates
            top_indices = torch.topk(torch.tensor(new_probabilities[b]), num_beams, largest=True).indices
            candidates[b] = [new_candidates[b][i] for i in top_indices]
            probabilities[b] = [new_probabilities[b][i] for i in top_indices]

    # Decode the generated candidates for each text in the batch
    batch_results = []
    for b in range(batch_size):
        next_texts = [tokenizer.decode(candidate) for candidate in candidates[b]]
        batch_results.append(list(zip(next_texts, candidates[b], probabilities[b])))

    return batch_results




def batch_custom_signle_search(model, tokenizer, device, texts, layer_ids, num_beams=3):

    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the input texts as a batch
    input_ids = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)

    # Initial single token candidates and probabilities
    with torch.no_grad():
        outputs = model(input_ids)

        top_indices_layers = {}
        top_probs_layers = {}
        for layer_id in layer_ids:
            layer_output = model.model.layers[layer_id].output
            normed = model.model.norm(layer_output)

            # Logits = [batch_size, tokens, vocab]
            logits = model.lm_head(normed)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, num_beams, dim=1)

            top_indices_layers[layer_id] = top_indices
            top_probs_layers[layer_id] = top_probs

    batch_results_layers = {}
    for layer_id in layer_ids:
        # Initialize candidates and probabilities for each text in the batch
        batch_size = input_ids.size(0)
        candidates = [[ [top_indices_layers[layer_id][b, i].item()] for i in range(num_beams) ] for b in range(batch_size)]
        probabilities = [[ top_probs_layers[layer_id][b, i].item() for i in range(num_beams) ] for b in range(batch_size)]

        # Decode the generated candidates for each text in the batch
        batch_results = []
        for b in range(batch_size):
            next_texts = [tokenizer.decode(candidate) for candidate in candidates[b]]
            batch_results.append(list(zip(next_texts, candidates[b], probabilities[b])))

        batch_results_layers[layer_id] = batch_results

    return batch_results_layers