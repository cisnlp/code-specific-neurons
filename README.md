# code-llm-lens

## Look Ahead in Logit Lens

In this work, we use the logit lens[^1] rather than the tuned lens[^2]. The tuned lens would undermine our goal of understanding whether the models, when prompted with `X`, take a detour through `Y` internal states before outputting the `X` text. Since the tuned lens is specifically trained to map internal states to the final `X` next-token prediction, it eliminates our signal of interest.

#### Logit Lens

Consider a pre-LayerNorm transformer model $\mathcal{M}$ that consists of multiple layers. We know that some vectors in embedding space make sense when converted into vocabulary space:

- The very first embedding vectors are just the input tokens.
- The very last embedding vectors are just the output logits.

What about the embedding vectors of the $l$ th layer? Logit Lens shows how hidden states at each layer contribute to the final token predictions. 

Transformer model $\mathcal{M}$ can be divided into two sections:
- $\mathcal{M}_{\leq \ell}$: This portion includes all layers up to and including layer $l$, which maps input tokens to hidden states.
- $\mathcal{M}_{>\ell}$: This portion encompasses all layers after $l$, which convert hidden states into logits.

The update mechanism for a transformer layer at index $l$ is given by:


```math
\mathbf{h}_{\ell+1} = \mathbf{h}_{\ell} + F_{\ell}(\mathbf{h}_{\ell}),
```

where $F_{\ell}$ is the residual output of layer $l$. 
If we do this recursively then we have:

```math
\mathcal{M}_{>\ell}(\mathbf{h}_{\ell}) = \mathrm{LayerNorm}\left[\mathbf{h}_{\ell} + \sum_{\ell'=\ell}^{L} F_{\ell'}(\mathbf{h}_{\ell'})\right] W_U,
```

where $W_U$ is the unembedding matrix that transforms the final hidden states into logits.
The core idea of the Logit Lens is to set the residual updates to zero, focusing solely on the hidden state from layer $l$:

```math
\mathrm{LogitLens}(\mathbf{h}_{\ell}) = \mathrm{LayerNorm}[\mathbf{h}_{\ell}] W_U.
```

#### How to look Ahead ?


[^1]: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens  
[^2]: https://github.com/AlignmentResearch/tuned-lens
