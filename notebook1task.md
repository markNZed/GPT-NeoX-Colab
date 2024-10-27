**Project Specification for Freelancer: Creating a GPT-NeoX Training Notebook on Shakespeare Dataset**

---

**Objective:**

Develop an introductory Colab notebook that demonstrates a straightforward setup for training a small GPT-NeoX model on the Shakespeare dataset. This notebook will serve as an entry point for exploring GPT-NeoX model training, tailored for use on a T4 GPU in Google Colab. The goal is to create a lightweight, educational environment where team members can become familiar with the GPT-NeoX setup, configuration, and basic training workflow.

---

### **Key Requirements:**

1. **Simple Model Setup:**
   - Configure a small GPT-NeoX model that can be easily trained on a T4 GPU.
   - Keep the model parameters low to ensure training runs quickly and fits within Colab’s resource limits.

2. **Text Data Preparation (Shakespeare):**
   - Utilize the Shakespeare text dataset to train the model.
   - Demonstrate basic data preparation techniques.

3. **Colab-Focused Setup:**
   - Ensure the notebook is designed for easy execution in Google Colab with clear, beginner-friendly explanations.
   - Provide modular code sections to facilitate learning and exploration.

4. **Collaboration and Experiment Tracking:**
   - Use GitHub for code versioning and sharing.
   - Integrate Weights & Biases (W&B) for logging and tracking metrics.

---

### **Notebook Structure and Content:**

This notebook should guide the user step-by-step through setting up, configuring, and training a simple GPT-NeoX model on the Shakespeare dataset, with clear explanations of each section.

#### **1. Introduction:**

- **Objective Statement:**
  - Briefly explain the notebook’s goal: setting up and training a small GPT-NeoX model on the Shakespeare dataset.

- **Overview of GPT-NeoX and its Capabilities:**
  - Provide a short introduction to GPT-NeoX, with emphasis on why it’s a suitable model for language tasks.

#### **2. Setup and Environment Configuration:**

- **Clone the GPT-NeoX Repository:**
  - Include commands to clone the GPT-NeoX GitHub repository.

  ```bash
  !git clone https://github.com/EleutherAI/gpt-neox.git
  %cd gpt-neox
  ```

- **Install Dependencies:**
  - Provide commands to install all necessary packages.

  ```bash
  !pip install -r requirements/requirements.txt
  !pip install -e .
  ```

- **Verify GPU Availability:**
  - Include a quick check to confirm that Colab has allocated a T4 GPU.

  ```python
  import torch
  print(torch.cuda.get_device_name(0))
  ```

#### **3. Data Preparation (Shakespeare Dataset):**

- **Download and Preprocess Data:**
  - Download the Shakespeare dataset (e.g., from [Karpathy’s Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)).
  - Split the text into sequences of a fixed length to prepare it for training.

  ```python
  import requests
  from pathlib import Path

  # Download Shakespeare data
  url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
  response = requests.get(url)
  Path("shakespeare.txt").write_text(response.text)

  # Split text into chunks
  with open("shakespeare.txt") as f:
      text = f.read()

  chunk_size = 256
  chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

  # Save as JSONL format
  with open("shakespeare.jsonl", "w") as f:
      for chunk in chunks:
          f.write(f'{{"text": "{chunk}"}}\n')
  ```

- **Tokenization:**
  - Train a custom tokenizer on the Shakespeare data to create a vocabulary suited for this task.

  ```python
  from tokenizers import ByteLevelBPETokenizer

  tokenizer = ByteLevelBPETokenizer()
  tokenizer.train(files=["shakespeare.txt"], vocab_size=5000, min_frequency=2, special_tokens=[
      "<s>",
      "<pad>",
      "</s>",
      "<unk>",
      "<mask>",
  ])
  tokenizer.save_model("tokenizer")
  ```

#### **4. Model Configuration:**

- **Define a Small GPT-NeoX Configuration:**
  - Use a simple configuration file with reduced parameters to ensure training fits on a T4 GPU.
  - Keep the model small to allow quick training.

  **Example Configuration Parameters:**

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
  optimizer:
    type: Adam
    lr-schedule: cosine
    warmup: 0.01
  data:
    data-path: "shakespeare.jsonl"
    seq-length: 256
  fp16:
    enabled: true
  logging:
    log-interval: 100
  checkpointing:
    save: true
    save-interval: 500
    save-dir: "checkpoints/"
  ```

- **Explanation of Config Parameters:**
  - Provide an overview of the chosen parameters and why they’re suitable for this task.

#### **5. Training with W&B Integration:**

- **Set Up W&B for Experiment Tracking:**
  - Install and configure Weights & Biases (W&B) to log metrics and track the training process.

  ```bash
  !pip install wandb
  ```

  ```python
  import wandb
  wandb.login()
  ```

- **Run the Training Loop:**
  - Use GPT-NeoX’s training script with W&B integration for logging.
  - Log training metrics and artifacts to W&B for easy tracking.

  ```python
  !python deepy.py train.py \
      --conf_dir=configs \
      --conf_file=shakespeare_config.yml \
      --wandb_project='shakespeare-training'
  ```

#### **6. Basic Experimentation:**

- **Experiment with Hyperparameters:**
  - Demonstrate how to adjust parameters like learning rate, batch size, and model depth.
  - Include explanations for each hyperparameter's effect.

- **W&B Tracking for Comparison:**
  - Use W&B’s experiment tracking features to log and compare different runs.

#### **7. Evaluation and Analysis:**

- **Evaluation Metrics:**
  - Implement basic evaluation metrics such as perplexity to gauge model performance.

- **Analyze Results with W&B:**
  - Guide users on how to view and interpret W&B metrics and visualizations.

#### **8. Save and Export the Model:**

- **Save the Model Checkpoints:**
  - Explain how to save model checkpoints to resume training if needed.

- **Load and Test the Model:**
  - Provide sample code to reload the model and generate Shakespeare-style text.

#### **9. GitHub and Collaboration Setup:**

- **GitHub Repository Structure:**
  - Organize the repository for collaborative development.
  
    ```
    /root
    ├── notebooks
    │   └── shakespeare_training_notebook.ipynb
    ├── configs
    │   └── shakespeare_config.yml
    ├── data
    │   └── (Data preparation instructions)
    ├── tokenizer
    │   └── (Tokenizer files)
    ├── README.md
    ├── .gitignore
    └── requirements.txt
    ```

- **GitHub Usage:**
  - Set up a clear `README.md` with setup instructions.
  - Use GitHub issues or discussions for collaboration and feedback.

---

### **Expectations for the Freelancer:**

- **Deliverables:**
  - A Colab notebook (`shakespeare_training_notebook.ipynb`) that meets the above specifications.
  - A GitHub repository with configuration files, tokenizer, and data preparation scripts.
  - Clear instructions and comments in the notebook for users new to GPT-NeoX.

- **Code Quality:**
  - Write clean, organized code with comments to explain each step.
  - Ensure reproducibility by documenting versions and setting random seeds.

- **Collaboration Setup:**
  - Use GitHub for code sharing and version control.
  - Ensure W&B logging is implemented to enable experiment comparison.

- **Communication:**
  - Regularly update on progress and request feedback if necessary.
  - Be open to revisions and adjustments based on team feedback.

---

### **Additional Considerations:**

- **Data Privacy and Licensing:**
  - Ensure the dataset and scripts comply with licensing and data privacy requirements.

- **Testing:**
  - Test the notebook thoroughly to ensure it runs smoothly end-to-end on a T4 GPU.

- **User Experience:**
  - Structure the notebook for a smooth learning experience, including error handling and debugging tips where needed.

### **W&B Integration for Collaboration:**

- **Project Setup:**
  - Create a W&B project specifically for this task, shared with the team.

- **Experiment Tracking:**
  - Log hyperparameters, metrics, and artifacts.
  - Use W&B’s comparison features to track progress across different training runs.

---

**Conclusion:**

This notebook will serve as an accessible, entry-level guide to training GPT-NeoX on the Shakespeare dataset, focusing on simplicity and clarity. By leveraging GitHub for collaboration and W&B for experiment tracking, this project aims to provide an interactive

, exploratory environment for learning and experimenting with language model training.
