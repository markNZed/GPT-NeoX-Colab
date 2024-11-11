import os
import subprocess
import time
import json
import re
import shutil
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator  # type: ignore
from transformers import GPTNeoXForCausalLM  # type: ignore
import torch

# Dataset Utilities
def text2jsonl(input_txt_file, output_jsonl_file):
    """
    Reads a text file and writes a JSONL file in the required format for
    training.
    """
    lines = []
    with open(input_txt_file, encoding="utf8") as f:
        for line in f.read().splitlines():
            if line:
                lines.append({"text": line})
    json_lines = [json.dumps(data) for data in lines]
    with open(output_jsonl_file, "w") as f:
        f.write("\n".join(json_lines))


def tokenize_dataset(
    input_file,
    output_prefix,
    tokenizer_type="CharLevelTokenizer",
    dataset_impl="mmap",
    append_eod=True,
    env_activate_cmd="source /content/my_env/bin/activate",
):
    """
    Tokenizes the dataset using GPT-NeoX's `preprocess_data.py` script.

    Parameters:
    - input_file: Path to the input JSONL dataset file.
    - output_prefix: Prefix for the output tokenized dataset.
    - tokenizer_type: Type of tokenizer (default is 'CharLevelTokenizer').
    - dataset_impl: Dataset implementation type (e.g., 'mmap').
    - append_eod: Whether to append end-of-document tokens (default is True).
    - env_activate_cmd: Command to activate the Python virtual environment.
    """
    import subprocess
    import os

    # Ensure input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the command to run the preprocess_data.py script
    tokenization_cmd = (
        f"{env_activate_cmd} && "
        f"cd /content/gpt-neox && "
        f"python tools/datasets/preprocess_data.py "
        f"--input {input_file} "
        f"--output-prefix {output_prefix} "
        f"--tokenizer-type {tokenizer_type} "
        f"--dataset-impl {dataset_impl} "
        f"{'--append-eod' if append_eod else ''}"
    )

    # Run the tokenization command
    subprocess.run(tokenization_cmd, shell=True, check=True)
    print(f"Dataset tokenized successfully. Output saved with prefix '{output_prefix}'")


# Training Utilities
def start_training(
    config_files, env_activate_cmd="source /content/my_env/bin/activate"
):
    """
    Starts the training process in the background and saves the process ID.
    """
    cmd = (
        'nohup bash -c "'
        f"{env_activate_cmd} && "
        "python ./deepy.py train.py --conf_dir /content/GPT-NeoX-Colab/configs "
        f"{' '.join(config_files)}\" > /dev/null 2>&1 & "
        "echo $! > train_process.pid"
    )
    subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)


def monitor_training(log_file, pid_file):
    """
    Monitors the training process, reads iterations from the log file, and
    checks if the process is running.
    """
    file_position = 0
    iteration_pattern = re.compile(r"iteration\s+(\d+)\s*/\s*\d+")

    def is_process_running(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    with open(pid_file, "r") as f:
        pid = int(f.read().strip())

    while is_process_running(pid):
        with open(log_file, "r") as file:
            file.seek(file_position)
            new_lines = file.readlines()
            file_position = file.tell()
            for line in new_lines:
                match = iteration_pattern.search(line)
                if match:
                    print(f"Iteration: {int(match.group(1))}")
        time.sleep(30)
    print("Training has finished.")


def wait_for_directory(directory_path, check_interval=10):
    """
    Waits until a specified directory is created.
    """
    while not os.path.exists(directory_path):
        print(f"Waiting for {directory_path} to be created...")
        time.sleep(check_interval)
    print(f"{directory_path} found.")


# Plot Utilities
def plot_training_and_validation_loss(tensorboard_log_dir):
    """
    Reads the TensorBoard logs and plots training and validation loss.
    """
    log_files = [
        os.path.join(tensorboard_log_dir, d) for d in os.listdir(tensorboard_log_dir)
    ]
    latest_log_dir = max(log_files, key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(latest_log_dir)
    ea.Reload()
    train_loss = ea.Scalars("train/lm_loss")
    val_loss = ea.Scalars("validation/lm_loss")
    train_loss_values = [x.value for x in train_loss]
    val_loss_values = [x.value for x in val_loss]
    iterations = range(1, len(train_loss_values) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_loss_values, label="Training Loss")
    plt.plot(iterations, val_loss_values, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


# Inference Utilities
def convert_model_to_huggingface(checkpoint_path, config_file, output_dir):
    """
    Converts the NeoX model checkpoint to HuggingFace format.
    """
    cmd = (
        "python ./tools/ckpts/convert_neox_to_hf.py "
        f"--input_dir {checkpoint_path} "
        f"--config_file {config_file} "
        f"--output_dir {output_dir} "
        "--precision auto "
        "--architecture neox"
    )
    subprocess.run(cmd, shell=True)


def generate_text_with_hf_model(model_path, tokenizer, input_text):
    """
    Generates text using the HuggingFace model and a given tokenizer.
    """
    model = GPTNeoXForCausalLM.from_pretrained(model_path)
    model.eval()
    input_ids = torch.tensor([tokenizer.tokenize(input_text)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=model.config.eos_token_id,
            do_sample=True,
        )
    generated_text = tokenizer.detokenize(output[0].tolist())
    print("Generated text:", generated_text)


# General Utilities
def get_latest_checkpoint(checkpoints_dir):
    """
    Reads the 'latest' file in the checkpoints directory to get the latest
    checkpoint path.
    """
    with open(os.path.join(checkpoints_dir, "latest"), "r") as f:
        latest_checkpoint_name = f.read().strip()
    return os.path.join(checkpoints_dir, latest_checkpoint_name)


# Drive Backup
def save_checkpoints_to_drive(source_folder, dest_folder):
    """
    Copies checkpoints to Google Drive, adding a version suffix if needed.
    """

    def get_versioned_folder_path(base_path):
        version = 1
        new_path = base_path
        while os.path.exists(new_path):
            new_path = f"{base_path}_v{version}"
            version += 1
        return new_path

    dest_versioned = get_versioned_folder_path(dest_folder)
    shutil.copytree(source_folder, dest_versioned)
    print(f"Folder copied successfully to Google Drive as '{dest_versioned}'!")

def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id