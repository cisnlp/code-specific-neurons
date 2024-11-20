"""
Visualize and Generate Heatmaps of Logit Lens

Copyright:
Amir Hossein Kargaran

License: Repository License.

Change Log:
- V 1.0: here.
"""



from code_lens.utils.beamsearch import custom_beam_search, batch_custom_beam_search, batch_custom_signle_search
import plotly.graph_objects as go
import numpy as np


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


def generate_heatmap(model, tokenizer, device, text, layers = [0], num_beams=1, max_length=2, min_position = None, max_position = None, batch_size=4):

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
                layer_batch_results = batch_custom_signle_search(model=model.model, tokenizer=tokenizer, device=device, texts=batch_temp_prompt, num_beams=num_beams, layer_ids=layers)

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
        title="LogitLens",
        xaxis_title="Tokens",
        yaxis_title="Layers",
        xaxis=dict(tickmode="array", tickvals=np.arange(len(token_indices_to_show)), ticktext=[str(output_tokens[t_id]) for index, t_id in enumerate(token_indices_to_show [:-1])], tickangle=90),
        yaxis=dict(tickmode="array", tickvals=np.arange(len(layers)), ticktext=[f"{i}" for i in layers]),
        hovermode='closest',
        autosize=True
    )

    return fig
