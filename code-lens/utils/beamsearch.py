"""
Beam Search

Description:
This code performs beam search for the next sequence of tokens given the intermediate layer_id.

Copyright:
Chengzhi Zhong (https://github.com/Sayn-Wittgenstein)

License: No explicit license provided.

Change Log:
- V 1.0: https://github.com/Sayn-Wittgenstein/latent_language_of_multilingual_model/blob/main/Translation.ipynb
- V 1.1: https://github.com/cisnlp/code-llm-lens/blob/main/beamsearch.py
"""


import torch
import torch.nn.functional as F

def custom_beam_search(model, tokenizer, device, text, layer_id, num_beams=5, max_length=50):
    
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
