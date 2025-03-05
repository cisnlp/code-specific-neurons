"""
Visualize and Generate Heatmaps of Logit Lens.
This code uses beam search for the next sequence of tokens given the intermediate layer_id.

Copyright:
Chengzhi Zhong (https://github.com/Sayn-Wittgenstein)
Amir Hossein Kargaran

License: Repository License.

Change Log:
- V 1.0: https://github.com/Sayn-Wittgenstein/latent_language_of_multilingual_model/blob/main/Translation.ipynb
- V 2.0: here.
"""

import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import numpy as np
import glob
import json



def get_code_keywords(path):
    # Get all file paths from the specified directory
    code_paths = glob.glob(path + '/*.json')  # Added '.json' to filter JSON files

    code_jsons = {}
    for code_path in code_paths:
        with open(code_path, 'r') as file:  # Open the file to read its content
            code_json = json.load(file)  # Load the JSON content

        # Extract the name and keywords/builtins
        code_name = code_json.get('name', 'default')
        code_jsons[code_name] = code_json.get('keywords', []) + code_json.get('builtins', [])

    return code_jsons


def trunc_string(string: str, new_len: int) -> str:
    """Truncate a string."""

    if new_len >= len(string):
        return  string 

    else:
        return  string[:new_len] + '…' # → '[…]'


def escape_html_tags(text):
    # Replace '<' with '&lt;' and '>' with '&gt;'
    escaped_text = text.replace("<", "&lt;").replace(">", "&gt;")
    return escaped_text



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




def batch_custom_single_search(model, tokenizer, device, texts, layer_ids, num_beams=3):

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


def generate_heatmap(model, tokenizer, device, text, layers = [0], num_beams=1, max_length=2, min_position = None, max_position = None, batch_size=1):

    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    tokens = tokens.flatten()

    if max_position is None:
        max_position = len(tokens)

    if min_position is None:
        min_position = 0

    min_position = min(min_position, len(tokens))
    max_position = max(max(0, max_position), min_position)


    heatmap_data = {
        'tokens': [tokenizer.decode([tokens[t].item()]) for t in range(tokens.size(0))],
        'layers': layers,
        'values': {layer: {i: [] for i in range(min_position, max_position)} for layer in layers}  # Empty list for each layer and token
    }


    if batch_size == 1 and max_length > 1:
        # Loop through tokens and layers to populate the heatmap data
        for token_idx in range(min_position, max(max_position, 0)):
            
            temp_tokens = tokens[:token_idx]
            temp_prompt = tokenizer.decode(temp_tokens)
    
            for layer in layers:
                # Perform beam search to get token results and their probabilities
                results = custom_beam_search(model=model.model, tokenizer=tokenizer, device=device, text=temp_prompt, num_beams=num_beams, max_length=max_length, layer_id=layer)
    
                # Collect the top tokens and their probabilities
                for result, id, probability in results:
                    heatmap_data['values'][layer][token_idx].append((result, probability))  # Append the result to the values list


    else:
        temp_prompts = []
        token_idxs = []
        for idx, token_idx in enumerate(range(min_position, max(max_position, 0))):
                
            temp_tokens = tokens[:token_idx]
            temp_prompt = tokenizer.decode(temp_tokens)
        
            if idx % batch_size == 0:
                temp_prompts.append([])
                token_idxs.append([])
        
            temp_prompts[-1].append(temp_prompt)
            token_idxs[-1].append(token_idx)
        
        if max_length==1:
            
            for batch_token_idx, batch_temp_prompt in zip(token_idxs, temp_prompts):
                # Perform beam search to get token results and their probabilities
                # TODO: For some models seems not working
                layer_batch_results = batch_custom_single_search(model=model.model, tokenizer=tokenizer, device=device, texts=batch_temp_prompt, num_beams=num_beams, layer_ids=layers)

                for layer in layers:
                    batch_results = layer_batch_results[layer]
                    # Collect the top tokens and their probabilities
                    for idx, results in enumerate(batch_results):
                        token_idx = batch_token_idx[idx]
                        for result, id, probability in results:
                            heatmap_data['values'][layer][token_idx].append((result, probability))  # Append the result to the values list

        elif max_length > 1:
            for layer in layers:
            
                for batch_token_idx, batch_temp_prompt in zip(token_idxs, temp_prompts):
                    # Perform beam search to get token results and their probabilities
                    batch_results = batch_custom_beam_search(model=model.model, tokenizer=tokenizer, device=device, texts=batch_temp_prompt, num_beams=num_beams, max_length=max_length, layer_id=layer)
            
                    # Collect the top tokens and their probabilities
                    for idx, results in enumerate(batch_results):
                        token_idx = batch_token_idx[idx]
                        for result, id, probability in results:
                            heatmap_data['values'][layer][token_idx].append((result, probability))  # Append the result to the values list

    return heatmap_data




def visualize_heatmap(heatmap_data, layers_to_show, token_indices_to_show, trunc_size = 6):
    """
    Visualizes a heatmap based on the provided heatmap data, tokens, and layers.

    Parameters:
    - heatmap_data (dict): A dictionary containing tokens, layers, and their respective probabilities.

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly figure containing the generated heatmap.
    """
    # Create the heatmap matrix (top_prob values) and annotations (top_values)
    heatmap_matrix = []
    annotations = []
    layers = layers_to_show

    output_tokens = heatmap_data['tokens'].copy()
    output_tokens.insert(0, '')


    for layer in layers:
        row = []
        for token_id in token_indices_to_show:
            # Get the top 3 tokens and their probabilities for this token-layer pair
            values = heatmap_data['values'][layer][token_id]
            top_values = sorted(values, key=lambda x: x[1], reverse=True)  # Sort by probability
            top1_prob = top_values[0][1]

            # Color the cell based on top1_prob
            row.append(top1_prob)

            # Prepare the annotation to show the top token
            annotation_text = top_values[0][0]
            annotation_text = escape_html_tags(annotation_text)
            annotation_text = trunc_string(annotation_text, trunc_size)

            annotations.append(
                dict(
                    x=token_indices_to_show.index(token_id),
                    y=layers.index(layer),
                    text=annotation_text, 
                    showarrow=False,
                    font=dict(size=10, color="white" if top1_prob >= 0.7 or top1_prob <= 0.3 else "black"),
                    align="center"
                )
            )

        heatmap_matrix.append(row)

    # Define the heatmap, where each cell displays the first token (top token) with color based on top1_prob
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=np.arange(len(token_indices_to_show)),
        y= np.arange(len(layers)),
        colorscale='rdbu',  # Color scale based on top1_prob
        colorbar=dict(title="Probability", titleside="right"),
        zmin=0,
        zmax=1,
        hovertemplate="%{text}<extra></extra>",  # Shows the top 3 tokens and their probabilities on hover
        text=[['' for _ in token_indices_to_show] for _ in layers],  # Default empty text for each cell
    ))

    # Update the cell text to show only the first token (from sorted top_values)
    for layer in layers:
        for token_id in token_indices_to_show:
            values = heatmap_data['values'][layer][token_id]
            top_values = sorted(values, key=lambda x: x[1], reverse=True)  # Sort by probability
            
            fig.data[0].text[layers.index(layer)][token_indices_to_show.index(token_id)] = "<br>".join([f"{escape_html_tags(t[0])}: {t[1]:.2f}" for t in top_values])

    # Add annotations for top 3 tokens
    for ann in annotations:
        fig.add_annotation(ann)

    # Update layout for better visualization
    fig.update_layout(
        title="Code-Lens",
        xaxis_title="Tokens",
        yaxis_title="Layers",
        xaxis=dict(tickmode="array", tickvals=np.arange(len(token_indices_to_show)), ticktext=[str(output_tokens[t_id]) for index, t_id in enumerate(token_indices_to_show [:-1])], tickangle=90),
        yaxis=dict(tickmode="array", tickvals=np.arange(len(layers)), ticktext=[f"{i}" for i in layers]),
        hovermode='closest',
        autosize=True
    )

    return fig


def update_top_values_with_majority(top_values, code_json):
    updated_top_values = []

    for original_string, score in top_values:
        # Split the string into components (words)
        components = original_string.split()
        match_counts = {}

        # Count matches for each code name
        for name, keywords in code_json.items():
            match_counts[name] = sum(1 for component in components if component in keywords)

        # Find code names where the majority of components match
        total_components = len(components)
        matching_code_names = [
            name for name, count in match_counts.items() if count / total_components >= 0.5
        ]

        # Update the string with matched code names if any are found
        if matching_code_names:
            updated_string = f"{original_string} [{' '.join(matching_code_names)}]"
        else:
            updated_string = original_string

        # Add the updated tuple to the new list
        updated_top_values.append((updated_string, score))

    return updated_top_values



def visualize_heatmap_code(heatmap_data, layers_to_show, token_indices_to_show, trunc_size = 6, keywords_path='../datasets/keywords'):
    """
    Visualizes a heatmap based on the provided heatmap data, tokens, and layers.

    Parameters:
    - heatmap_data (dict): A dictionary containing tokens, layers, and their respective probabilities.

    Returns:
    - fig (plotly.graph_objects.Figure): A Plotly figure containing the generated heatmap.
    """
    ## Init keywords
    code_jsons = get_code_keywords(path=keywords_path)


    # Create the heatmap matrix (top_prob values) and annotations (top_values)
    heatmap_matrix = []
    annotations = []
    layers = layers_to_show

    output_tokens = heatmap_data['tokens'].copy()
    output_tokens.insert(0, '')


    for layer in layers:
        row = []
        for token_id in token_indices_to_show:
            # Get the top 3 tokens and their probabilities for this token-layer pair
            values = heatmap_data['values'][layer][token_id]
            top_values = sorted(values, key=lambda x: x[1], reverse=True)  # Sort by probability
            
            top_values = update_top_values_with_majority(top_values, code_jsons)

            top1_prob = top_values[0][1]

            # Color the cell based on top1_prob
            row.append(top1_prob)

            # Prepare the annotation to show the top token
            annotation_text = top_values[0][0]



            annotation_text = escape_html_tags(annotation_text)
            annotation_text = trunc_string(annotation_text, trunc_size)

            annotations.append(
                dict(
                    x=token_indices_to_show.index(token_id),
                    y=layers.index(layer),
                    text=annotation_text, 
                    showarrow=False,
                    font=dict(size=10, color="white" if top1_prob >= 0.7 or top1_prob <= 0.3 else "black"),
                    align="center"
                )
            )

        heatmap_matrix.append(row)

    # Define the heatmap, where each cell displays the first token (top token) with color based on top1_prob
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=np.arange(len(token_indices_to_show)),
        y= np.arange(len(layers)),
        colorscale='rdbu',  # Color scale based on top1_prob
        colorbar=dict(title="Probability", titleside="right"),
        zmin=0,
        zmax=1,
        hovertemplate="%{text}<extra></extra>",  # Shows the top 3 tokens and their probabilities on hover
        text=[['' for _ in token_indices_to_show] for _ in layers],  # Default empty text for each cell
    ))

    # Update the cell text to show only the first token (from sorted top_values)
    for layer in layers:
        for token_id in token_indices_to_show:
            values = heatmap_data['values'][layer][token_id]
            top_values = sorted(values, key=lambda x: x[1], reverse=True)  # Sort by probability
            
            fig.data[0].text[layers.index(layer)][token_indices_to_show.index(token_id)] = "<br>".join([f"{escape_html_tags(t[0])}: {t[1]:.2f}" for t in top_values])

    # Add annotations for top 3 tokens
    for ann in annotations:
        fig.add_annotation(ann)

    # Update layout for better visualization
    fig.update_layout(
        title="Code-Lens",
        xaxis_title="Tokens",
        yaxis_title="Layers",
        xaxis=dict(tickmode="array", tickvals=np.arange(len(token_indices_to_show)), ticktext=[str(output_tokens[t_id]) for index, t_id in enumerate(token_indices_to_show [:-1])], tickangle=90),
        yaxis=dict(tickmode="array", tickvals=np.arange(len(layers)), ticktext=[f"{i}" for i in layers]),
        hovermode='closest',
        autosize=True
    )

    return fig
