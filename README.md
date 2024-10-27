# DRAFT/EXPERIMENTAL - DO NOT USE

# GPT-NeoX-Colab

> **An accessible set of Google Colab notebooks for training and experimenting with GPT-NeoX models on limited resources.**


## Overview

This repository provides a collection of example Google Colab notebooks that guide users through setting up and training GPT-NeoX models on tasks such as text generation and code completion. Tailored for ease of use, these notebooks run efficiently on single consumer-grade GPUs (like a T4 on Colab), providing an educational environment for experimenting with GPT-NeoX’s configurations, data handling, and training workflows.

## Features

- **Lightweight Model Configurations** – Optimized settings for fast training on single GPUs.
- **Custom Data Handling** – Instructions on loading, preprocessing, and tokenizing custom datasets.
- **Hyperparameter Experimentation** – Modular code to quickly adjust configurations and observe results.
- **Experiment Tracking with Weights & Biases (W&B)** – Track model metrics, hyperparameters, and checkpoints.
- **Collaboration-Ready** – GitHub-based code sharing with clear structure and collaborative tools.

## Available Notebooks

### 1. Shakespeare Text Generation

**Objective:** Train a small GPT-NeoX model on the Shakespeare dataset to explore basic text generation.

**Highlights:**
- Setup and configuration guidance for small, Colab-friendly models.
- Step-by-step instructions on data loading, tokenization, and model training.
- Integrated experiment tracking with W&B to log metrics and visualize model performance.

**Example Configuration:**
```yaml
model:
  hidden-size: 256
  num-layers: 4
  num-attention-heads: 4
  max-position-embeddings: 256
training:
  micro-batch-size: 2
  global-batch-size: 8
  train-iters: 5000
  lr: 5e-4
```

[**Open in Colab ➔**](link_to_shakespeare_notebook)

### 2. Python Code Completion

**Objective:** Train a GPT-NeoX model from scratch for Python code completion tasks.

**Highlights:**
- Includes dataset recommendations and preprocessing steps specific to Python code.
- Detailed sections on customizing training algorithms and hyperparameters.
- Integrated W&B tracking to facilitate comparison of different model configurations.

**Example Configuration:**
```yaml
hidden-size: 512
num-layers: 6
num-attention-heads: 8
max-position-embeddings: 1024
micro-batch-size: 1
global-batch-size: 8
train-iters: 10000
lr: 5e-4
```

[**Open in Colab ➔**](link_to_code_completion_notebook)

## Quick Start

### Prerequisites
- **Google Colab** – A free Google account with access to Colab GPUs.
- **GitHub Account** – For code sharing and collaboration.
- **Weights & Biases (W&B) Account** – For experiment tracking.

### Setup and Execution

1. **Clone the Repository**  
   Open a Colab notebook, then run:
   ```bash
   !git clone https://github.com/YourUsername/GPT-NeoX-Colab.git
   %cd GPT-NeoX-Colab
   ```

2. **Install Dependencies**  
   ```bash
   !pip install -r requirements.txt
   ```

3. **Verify GPU Availability**  
   ```python
   import torch
   print(torch.cuda.get_device_name(0))
   ```

4. **Run the Notebook**  
   Open the desired notebook ([Shakespeare](link_to_shakespeare_notebook) or [Code Completion](link_to_code_completion_notebook)) in Google Colab and follow the setup and training instructions.

## Repository Structure

```
GPT-NeoX-Colab/
├── notebooks/
│   ├── shakespeare_training_notebook.ipynb
│   └── code_completion_notebook.ipynb
├── configs/
│   ├── shakespeare_config.yml
│   └── code_completion.yml
├── data/
│   └── (Data instructions and files)
├── tokenizer/
│   └── (Tokenizer files)
├── scripts/
│   └── (Helper scripts for training and data processing)
├── README.md
├── .gitignore
└── requirements.txt
```

## Key Components

### 1. GitHub for Collaboration
- **Version Control:** Use GitHub for all code updates, bug tracking, and feature requests.
- **Clear Commit Messages:** Make changes with clear, descriptive messages to facilitate collaboration.
- **Repository Organization:** The repository is structured to keep configurations, scripts, and data handling in separate folders for clarity.

### 2. Weights & Biases (W&B) for Experiment Tracking
- **Project Creation:** Log into W&B and create a project to track experiments.
- **Automated Logging:** The notebooks are configured to log hyperparameters, metrics, and artifacts in real-time.
- **Comparisons:** Easily compare different model runs, configurations, and metrics.

## Additional Resources

- **GPT-NeoX Documentation:** [EleutherAI GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- **SourceGraph for Code Navigation:** [SourceGraph GPT-NeoX](https://sourcegraph.com/github.com/EleutherAI/gpt-neox) – For navigating the GPT-NeoX codebase.
- **Benchmarking Datasets:**  
  - [CodeXGLUE Token-Level Code Completion](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token)
  - [CodeXGLUE Line-Level Code Completion](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line)

## Contribution Guidelines

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a new branch (`feature/some-feature`).
3. Commit your changes and open a pull request.
4. Ensure that your code is well-documented and follows best practices.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- Special thanks to [EleutherAI](https://www.eleuther.ai/) for developing GPT-NeoX and providing the open-source community with invaluable tools and resources.
- Thanks to [Weights & Biases](https://wandb.ai/) for providing excellent tools for experiment tracking.

--- 

For further assistance in exploring the GPT-NeoX codebase, visit [SourceGraph GPT-NeoX](https://sourcegraph.com/github.com/EleutherAI/gpt-neox).
