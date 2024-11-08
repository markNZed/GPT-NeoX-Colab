import os
import subprocess
import time
import json
import re
import shutil
import traceback
import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator  # type: ignore
from transformers import GPTNeoXForCausalLM  # type: ignore
import torch
from dotenv import load_dotenv


def get_repo_path():
    return "/content/GPT-NeoX-Colab"


def load_env(dotenv_path=get_repo_path()):
    load_dotenv(dotenv_path=dotenv_path)
    print("Loaded environment variables from:", dotenv_path)


def is_colab():
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        return False


# Drive Utilities
def mount_google_drive(save_to_drive=False):
    """
    Mounts Google Drive.
    """
    if save_to_drive and is_colab():
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")


def setup_ssh_connection(ssh_connect=False):
    """
    Sets up an SSH connection to the Colab instance for remote access, such as from VS Code.
    """
    if ssh_connect and is_colab():
        # type: ignore
        from google.colab import userdata  # type: ignore

        import pexpect  # type: ignore
        import getpass

        if "SSH_CONNECTED" not in globals():
            SSH_CONNECTED = None

        # Check if SSH connection information is available
        if userdata.get("REMOTE_SSH") and not SSH_CONNECTED:
            SSH_CONNECTED = True
            print("Setting up SSH connection...")

            # Install OpenSSH server and set root password
            os.system("apt-get install -y openssh-server")
            os.makedirs("/var/run/sshd", exist_ok=True)
            os.system("echo 'root:root' | chpasswd")  # Set root password to 'root'
            os.system("echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config")
            os.system("service ssh restart")

            # Optional: install screen for background sessions
            os.system("apt-get install -y screen")

            # Prompt for SSH password and initiate SSH tunneling
            password = getpass.getpass("Enter your SSH password: ")
            ssh_command = (
                f"ssh -N -R 2223:localhost:22 -o StrictHostKeyChecking=no "
                f"-o ServerAliveInterval=60 -o ServerAliveCountMax=5 {userdata.get('REMOTE_SSH')}"
            )

            # Use pexpect to handle the SSH login prompt
            child = pexpect.spawn(ssh_command, encoding="utf-8")
            child.expect(["password:", "Password:"], timeout=60)
            child.sendline(password)
            child.sendline("")  # Empty command to keep session open

            print("SSH session is running in the background.")
            print("To connect from your local machine, use the following command:")
            print(f"ssh -L 9999:localhost:2223 {userdata.get('REMOTE_SSH')}")
    else:
        print(
            "This function requires Google Colab and won't work in other environments."
        )


# Backblaze Utilities
def get_b2_resource(endpoint, key_id, application_key):
    """
    Returns a boto3 resource object for B2 service.
    """
    return boto3.resource(
        service_name="s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key,
        config=Config(signature_version="s3v4"),
    )


def upload_file_to_b2(bucket, directory, file, b2, b2path=None):
    """
    Uploads a specified file to a Backblaze B2 bucket.
    """
    file_path = os.path.join(directory, file)
    remote_path = b2path if b2path else file
    try:
        response = b2.Bucket(bucket).upload_file(file_path, remote_path)
        return response
    except ClientError as ce:
        print("Upload error:", ce)
        traceback.print_exc()


def download_file_from_b2(bucket, directory, file, key_name, b2):
    """
    Downloads a specified object from Backblaze B2 and writes to local file
    system.
    """
    file_path = os.path.join(directory, file)
    try:
        b2.Bucket(bucket).download_file(key_name, file_path)
    except ClientError as ce:
        print("Download error:", ce)
        traceback.print_exc()


# Environment Setup
def download_venv(use_backblaze=False, b2_params=None):
    """
    Prepares the environment by downloading and unpacking the environment from
    Backblaze if not already present.
    """
    if not os.path.exists("/content/my_env.tar"):
        if use_backblaze and b2_params:
            b2 = get_b2_resource(**b2_params)
            download_file_from_b2(
                b2_params["bucket"], "/content", "my_env.tar.gz", "my_env.tar.gz", b2
            )
            # Add commands to unzip and untar as required


# Dataset Utilities
def prepare_custom_dataset(input_txt_file, output_jsonl_file):
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


def run(cmd, check=False):
    """Run a shell command and return its output."""
    # print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, check=check)
    if result.returncode != 0:
        raise Exception(f"Command failed with return code {result.returncode} {result}")
    return result


def install_git_annex():
    """Install git-annex if it's not available."""
    try:
        run("git-annex version")
        print("git-annex is already installed.")
    except Exception as e:
        print("git-annex not found. Installing...")
        run("apt-get update")
        run("apt-get install -y git-annex")


def enable_remote(repo_path=get_repo_path()):
    """Enable git annex backblaze remote."""
    os.chdir(repo_path)
    run("git annex enableremote backblaze")


def sync_and_get_data():
    """Sync git-annex and download data from backblaze with error handling."""
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    try:
        print("Starting git annex sync...")
        run("git annex sync --no-push")
        print("Sync successful. Fetching data from backblaze...")
        run("git annex get . --from=backblaze")
        print("Data retrieval successful.")
    except subprocess.CalledProcessError as e:
        print("Error during git annex sync or data retrieval.")
        print(f"Command output: {e.output.decode() if e.output else 'No output'}")
        print(f"Command error message: {e.stderr.decode() if e.stderr else 'No additional error info.'}")
        raise  # Re-raise the exception to signal failure


def upload_file(bucket, directory, file, b2, b2path=None):
    file_path = directory + '/' + file
    remote_path = b2path
    if remote_path is None:
        remote_path = file
    try:
        response = b2.Bucket(bucket).upload_file(file_path, remote_path)
    except ClientError as ce:
        print('error', ce)
        traceback.print_exc()  # Print the full stack trace
    return response


# Download the specified object from B2 and write to local file system
def download_file(bucket, directory, file, key_name, b2):
    file_path = directory + '/' + file
    try:
        b2.Bucket(bucket).download_file(key_name, file_path)
    except ClientError as ce:
        print('error', ce)
        traceback.print_exc()  # Print the full stack trace


def download_my_env(upload_env=False):
    if (upload_env):
        return
    b2_r = get_b2_resource(os.getenv("BB_ENDPOINT"), os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    if not os.path.exists("/content/my_env.tar"):
        # Downloading takes about 40sec
        print("Downloading my_env.tar.gz")
        download_file(os.getenv("BB_BUCKET"), "/content", "my_env.tar.gz", "my_env.tar.gz", b2_r)
        print("Unzipping my_env.tar.gz")
        run("pigz -d -p 4 /content/my_env.tar.gz")  # Decompress using 4 cores (adjust as needed)
        print("Untarring my_env.tar.gz")
        run("pv /content/my_env.tar | tar -xf - -C /content")


def upload_my_env(upload_env=False):
    if (not upload_env):
        return
    key_id_rw = "003cb130fbeaa800000000002"
    b2_rw = get_b2_resource(os.getenv("BB_ENDPOINT"), key_id_rw, userdata.get('B2_APP_KEY_RW'))
    print("my_env.tar not found, proceeding with compression.")
    # Change directory (if needed) to ensure paths are correct
    os.chdir("/content")
    # Compress the tar archive with pigz, showing a progress indicator
    # Compresses 5.5G to 3.1G
    run("time tar cf - my_env | pv -p -e -r -b | pigz -p 4 -1 > my_env.tar.gz")
    # Upload to BackBlaze
    if userdata.get('B2_APP_KEY_RW'):
        response = upload_file(os.getenv("BB_BUCKET"), "/content", "my_env.tar.gz", b2_rw)
        print("Upload response:", response)
    else:
        print("Skipping upload to BackBlaze because no B2_APP_KEY_RW")
