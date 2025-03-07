## code-specific-neurons

Code and data for the ```How Programming Concepts and Neurons Are Shared in Code Language Models``` paper.

```
git clone https://github.com/cisnlp/code-specific-neurons.git
```

### code-logitlens 

To interpret latent embeddings, we use the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). We implement our version of Logit Lens in `code-logitlens/compute_lens.ipynb`. It uses the [datasets/parallel](datasets/parallel) and perform the translation task from one programming langauge to another. It sets one language as the input language and the other as the output language. It records the decoded tokens for each token and layer along with their probabilities and ranks.

### code-mexa 
To calculate cross-lingual alignment between programming languages, we use MEXA.  
MEXA uses [datasets/parallel](datasets/parallel) to compute alignment between a pivot language and the other languages. We use the [MEXA codebase](https://github.com/cisnlp/MEXA) and implement our code in `code-mexa/compute_mexa.ipynb`.

### code-lape

To calculate language-specific neurons, we use LAPE. LAPE uses [datasets/raw](datasets/raw) to identify language-specific neurons within LLMs.  
We use the [LAPE codebase](https://github.com/rucaibox/language-specific-neurons). The majority of the code remains unchanged, but we add `code-lape/id-gen.ipynb`, which is missing from the original code, and modify `code-lape/identify.ipynb` to ensure the same number of neurons is selected for each language.

### datasets

**keywords**: We included keywords and built-ins for different programming languages in the [datasets/keywords](datasets/keywords). Built-ins include: primitive types, macros, modules, collections, containers, and built-in functions, excluding keywords.

**parallel**: We store parallel few-shot prompts in different languages in [datasets/parallel/prompts](datasets/parallel/prompts). The code to generate parallel data from the [MuST-CoST repository](https://github.com/reddy-lab-code-research/MuST-CoST) is in [datasets/parallel/download_parallel.ipynb](datasets/parallel/download_parallel.ipynb). For reproducibility, we include the results in [datasets/parallel/code_snippets.zip](datasets/parallel/code_snippets.zip) (Extract the ZIP file).

**raw**: The code to generate raw data from the [Codeparrot repository](https://huggingface.co/datasets/codeparrot/github-code) and Wikipedia is in [datasets/raw/download_raw.ipynb](datasets/raw/download_raw.ipynb).

### citation

If you find our method, code, and data useful for your research, please cite:

```bib
@article{kargaran2025programming,
  title={How Programming Concepts and Neurons Are Shared in Code Language Models},
  author={Kargaran, Amir Hossein and Liu, Yihong and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint},
  year={2025}
}
```
