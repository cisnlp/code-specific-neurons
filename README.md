# Code Specific Neurons

Code and data for the ```How Programming Concepts and Neurons Are Shared in Code Language Models``` paper.

```
git clone https://github.com/cisnlp/code-specific-neurons.git
```

## Datasets

#### Parallel Data

We store parallel few-shot prompts in different languages in [datasets/parallel/prompts](datasets/parallel/prompts). The code to generate parallel data from the [MuST-CoST repository](https://github.com/reddy-lab-code-research/MuST-CoST) is in [datasets/parallel/download_parallel.ipynb](datasets/parallel/download_parallel.ipynb).  

For reproducibility, we include the results in [datasets/parallel/code_snippets.zip](datasets/parallel/code_snippets.zip). Extract the ZIP file.

#### Code Keywords and Built-ins

We included keywords and built-ins for different programming languages in the [datasets/keywords](datasets/keywords). Built-ins include: primitive types, macros, modules, collections, containers, and built-in functions, excluding keywords.

#### Raw

 The code to generate raw data from the [Codeparrot repository](https://huggingface.co/datasets/codeparrot/github-code) and Wikipedia is in [datasets/raw/download_raw.ipynb](datasets/parallel/download_raw.ipynb).


## Code-Logitlens 

To interpret latent embeddings, we use the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).

We implement our version of Logit Lens in `compute_lens.ipynb`. It uses the [parallel data](datasets/parallel) and sets one language as the input language and the other as the output language. It then performs translation tasks, recording the decoded tokens along with their probabilities and ranks.

## Code-MEXA 
To calculate cross-lingual alignment between programming languages, we use MEXA.

## Code-LAPE

To calculate language-specific neurons, we use LAPE.
