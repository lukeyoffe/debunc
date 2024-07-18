# DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations

This repo contains the code and data for the paper ["DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations"](https://arxiv.org/abs/2407.06426).



## Installation
```
git clone https://github.com/lukeyoffe/debunc.git
cd debunc
conda create --name debunc python=3.10 -y 
conda activate debunc 
pip install -e .
```

To use restricted models, log in to Hugging Face with the following command:
```
huggingface-cli login
```

## Usage
The scripts to run and evaluate on various benchmarks can be found in [src/debate/](src/debate/).

To use Llama 3 instead of Mistral 7B, replace `"mistralai/Mistral-7B-Instruct-v0.2"` with `"meta-llama/Meta-Llama-3-8B-Instruct"`. Other models are not currently supported.

To use TokenSAR instead of Mean Token Entropy, replace `ue_method = MeanTokenEntropy()` with `ue_method = TokenSAR()`

## Attention Scaling Demo
[src/models/demo.ipynb](src/models/demo.ipynb) contains a demonstration of attention scaling applied to RAG.

## Citation
```
@article{yoffe2024debunc,
  title={DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations},
  author={Yoffe, Luke and Amayuelas, Alfonso and Wang, William Yang},
  journal={arXiv preprint arXiv:2407.06426},
  year={2024}
}
```

## Acknowledgement
The code in [src/lm-polygraph](src/lm_polygraph/) is based on the  [LM-Polygraph](https://github.com/IINemo/lm-polygraph) project, and contains implementations for various uncertainty metrics.

The `modeling_*.py` files in [src/models](src/models/) are based on the Huggingface [Transformers](https://github.com/huggingface/transformers) library, with modifications to perform attention scaling. The attention scaling occurs between `##### <AttentionScaling> #####` and `##### </AttentionScaling> #####`.