# GPT-NeoX-Colab

> **An accessible set of Google Colab notebooks for training and experimenting with GPT-NeoX SLM models on limited resources.**

## Status

The Shakespeare Text Generation Notebook is available. The Python Code Completion Notebook is under development.

## Overview

This repository provides a collection of example Google Colab notebooks that guide users through setting up and training GPT-NeoX models on tasks such as text generation and code completion. Tailored for ease of use, these notebooks run efficiently on single consumer-grade GPUs (like a T4 on Colab), providing an educational environment for experimenting with GPT-NeoX’s configurations, data handling, and training workflows.

## Features

- **Lightweight Model Configurations** – Optimized settings for fast training on single GPUs.
- **Custom Data Handling** – Instructions on loading, preprocessing, and tokenizing custom datasets.
- **Hyperparameter Experimentation** – Modular code to quickly adjust configurations and observe results.
- **Experiment Tracking with DagsHub** – Track model metrics, hyperparameters, and checkpoints.
- **Collaboration-Ready** – GitHub-based code sharing with clear structure and collaborative tools.

## Available Notebooks

### 1. Shakespeare Text Generation

**Objective:** Train a small GPT-NeoX model on the Shakespeare dataset to explore basic text generation.

**Highlights:**

- Setup and configuration guidance for small, Colab-friendly models.
- Step-by-step instructions on data loading, tokenization, and model training.
- Integrated experiment tracking with DagsHub to log metrics and visualize model performance.

[**Open in Colab ➔**](notebooks/shakespeare_training.ipynb)

### 2. Python Code Completion

**Objective:** Train a GPT-NeoX model from scratch for Python code completion tasks.

**Highlights:**

- Includes dataset recommendations and preprocessing steps specific to Python code.
- Detailed sections on customizing training algorithms and hyper-parameters.
- Integrated DagsHub tracking to facilitate comparison of different model configurations.
- Evaluation on a public benchmark

[**Open in Colab ➔**](notebooks/code_completion_training.ipynb)

## Quick Start

### Prerequisites

- **Google Colab** – A free Google account with access to Colab GPUs.
- **GitHub Account** – For code sharing and collaboration.
- **DagsHub Account** – For experiment tracking.

### Setup and Execution

1. **Run the Notebook**

   From this GitHub repository open the Colab notebook in Colab using the Colab link at the top of the notebook.

   - [Shakespeare](notebooks/shakespeare_training.ipynb)
   - [Code Completion](notebooks/code_completion_training.ipynb)
   - Follow the setup and training instructions in the notebook

## Repository Structure

```
GPT-NeoX-Colab/
├── notebooks/
│   ├── shakespeare_training.ipynb
│   └── code_completion_training.ipynb
├── configs/
│   ├── shakespeare.yaml
│   └── code_completion.yaml
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

### 2. DagsHub for Experiment Tracking

- **Project Creation:** Log into DagsHub and create a project to track experiments.
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
- Thanks to [Weights &amp; Biases](https://wandb.ai/) for providing excellent tools for experiment tracking.

---

For further assistance in exploring the GPT-NeoX codebase, visit [SourceGraph GPT-NeoX](https://sourcegraph.com/github.com/EleutherAI/gpt-neox).
