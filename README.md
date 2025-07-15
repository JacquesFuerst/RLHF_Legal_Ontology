# Reinforcement Learning from Human Feedback (RLHF) for Legal Ontology Information Extraction

This project explores using RLHF and RLAIF to finetune a language model (Mistral-7B0Instruct-v0.3) using a small, domain specific dataset of 26 prompts in total (18 train, 4 eval, 4 test; about 3000 tokens each) from the Dutch legal domain. The goal is to evaluate how effective this method can be in a sparse data scenario to see whether it could be adapted in a similar setting, where possibly more resources of some dimension are available.

## Table of Contents
- [Installation- Usage
- Dataset
- Training Details
- Results
- Citation
- License



## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/your-repo.git #TODO: add my github repo where I push the code
cd your-repo
pip install -r requirements.txt
```

## Usage

To run inference:

```bash
python run_inference.py --model_path ./checkpoints/mistral-finetuned
```


# Data 

The dataset consists of several dimensions with ~3000 tokens each, designed to test the model's reasoning and domain adaptation. See `data/README.md` for details. 

TODO: write out details of each folder...


## Training Details

- Model: Mistral 7B Instruct v0.3
- GPUs: 3x L40S
- Optimizer: AdamW-8bit
- Epochs: 10
- Batch size: 2



## Results

| Metric        | Baseline | Fine-Tuned |
|---------------|----------|------------|
| Accuracy      | 72.5%    | 89.3%      |
| BLEU Score    | 0.41     | 0.58       |

See `results/` for full evaluation logs and plots.



## Citation

If you use this work, please cite:


@misc{yourname2025mistralfinetune, title={Fine-Tuning Mistral 7B on Domain-Specific Prompts}, author={Fürst, Jacques}, year={2025}, url={https://github.com/yourusername/your-repo} }


## License

This project is licensed under the MIT License. See `LICENSE` for details

