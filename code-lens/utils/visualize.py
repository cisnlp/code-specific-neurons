from code_lens.utils.beamsearch import custom_beam_search
import plotly.graph_objects as go
import numpy as np

def generate_heatmap(model, tokenizer, device, text, num_beams=1, max_length=2, layers = [0, 20], min_position = 123, max_position = 133):

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


    # Loop through tokens and layers to populate the heatmap data
    for token_idx in range(min_position, max(max_position, 0)):
        
        temp_tokens = tokens[:token_idx]
        temp_prompt = tokenizer.decode(temp_tokens)

        for layer in layers:
            # Perform beam search to get token results and their probabilities
            results = custom_beam_search(model=model.model, tokenizer=tokenizer, device=device, text=temp_prompt, num_beams=num_beams, max_length=max_length, layer_id=layer)

            # Collect the top 3 tokens and their probabilities
            for result, id, probability in results:
                heatmap_data['values'][layer][token_idx].append((result, probability))  # Append the result to the values list

    return heatmap_data



def visualize_heatmap(heatmap_data, layers_to_show, token_indices_to_show):
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
            annotations.append(
                dict(
                    x=token_indices_to_show.index(token_id),
                    y=layers.index(layer),
                    text=annotation_text, 
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center"
                )
            )

        heatmap_matrix.append(row)

    # Define the heatmap, where each cell displays the first token (top token) with color based on top1_prob
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix,
        x=np.arange(len(token_indices_to_show)),
        y= np.arange(len(layers)),
        colorscale='RdBu',  # Color scale based on top1_prob
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
            
            fig.data[0].text[layers.index(layer)][token_indices_to_show.index(token_id)] = "<br>".join([f"{t[0]}: {t[1]:.2f}" for t in top_values])

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
        autosize=False
    )

    return fig