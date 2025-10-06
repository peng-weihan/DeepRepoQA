# DeepRepoQA

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-green.svg)]()

## Introduction

**DeepRepoQA** is a framework for *deep repository question answering* in realistic code environments.

Effectively answering developer questions about a software repository is a critical yet under-explored problem in software engineering. While existing repository understanding methods have advanced the field, they predominantly rely on surface-level code retrieval and lack the ability for deep reasoning over multiple files, complex software architectures, and grounding answers in long-range code dependencies. 

To address these limitations, **DeepRepoQA** builds on the *agentic framework*, where LLM agents find answers through a systematic tree search over structured action spaces. Our key innovations include:
- Balanced exploration and exploitation via **Monte Carlo Tree Search (MCTS)** for multi-hop repository reasoning.
- **LLM feedback** that provides learned priors and values to reduce search depth and mitigate drift.
- **Structured memory paths** that enable reliable evidence synthesis and traceable reasoning steps.

Comprehensive experiments on **SWE-QA** demonstrate substantial performance gains over strong baselines, validating the effectiveness of systematic MCTS-guided exploration for multi-hop repository reasoning.

<p align="center">
  <img src="assets/approach.png" alt="Approach" style="max-width:80%; height:auto;"/>
</p>


DeepRepoQA is a project for deep repository question answering.


The benchmark dataset used in our experiments is available on Hugging Face:
- **Dataset**: [SWE-QA-Benchmark](https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark)

## Installation

### Requirements

- Python 3.11 (recommended to use conda environment)
- Other dependencies...

### Environment Setup

Recommended to use conda for creating an isolated environment:

```bash
# Create conda environment using environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate DeepRepoQA
```

## Quick Start

This project supports **single mode** and **batch mode**. Before running, make sure to configure the environment variables properly.

---

### Environment Configuration

Create a `.env` file in the project root and add the following:

```ini
# Custom LLM configuration
CUSTOM_LLM_API_BASE=""
CUSTOM_LLM_MODEL=""
CUSTOM_LLM_API_KEY=""

# Voyage API configuration (required)
VOYAGE_API_KEY=""

# Required for single mode
REPO_PATH=""
````

**Explanation of variables:**

* `CUSTOM_LLM_API_BASE`: API endpoint of your custom LLM
* `CUSTOM_LLM_MODEL`: Model name, e.g., `gpt-4`
* `CUSTOM_LLM_API_KEY`: API key for accessing the LLM
* `VOYAGE_API_KEY`: Required for using the Voyage API
* `REPO_PATH`: Only needed for **single mode**, specifies the repository path

---

### Single Mode

Run the single mode example:

```bash
python example.py
```

> This mode processes the repository specified in `REPO_PATH` in your `.env` file.

---

### Batch Mode

Run the batch mode example:

```bash
python example_batch.py
```

> This mode processes multiple repositories sequentially.

---

### Input

Questions are saved in:

```
dataset/questions
```

### Output

Results from batch mode are saved in:

```
dataset/answers
```

Each question is stored as a JSONL file with answers



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
