# DeepRepoQA

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-green.svg)]()

## Introduction

DeepRepoQA is a project for deep repository question answering.
![Approach](assets/approach.png)
## Features


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

### Output

Results from both single mode and batch mode are saved in:

```
dataset/answers
```

Each repository/question is stored as a JSONL file with answers



## Project Structure

```
```


## Contributing

Contributions are welcome! Please follow these steps:


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

## Acknowledgments

