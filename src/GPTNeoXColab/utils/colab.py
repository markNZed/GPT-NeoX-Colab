import os
import subprocess
import traceback
import boto3  # type: ignore
from botocore.config import Config  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from dotenv import load_dotenv
from pathlib import Path
try:
    from google.colab import userdata  # type: ignore
except ImportError:
    pass


def load_env():
    load_dotenv("/content/GPT-NeoX-Colab/.env")


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
            "This function requires Google Colab to get REMOTE_SSH and won't work in other environments."
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


def run(cmd, check=False):
    """Run a shell command and return its output."""
    # print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, check=check)
    if result.returncode != 0:
        raise Exception(f"Command failed with return code {result.returncode} {result}")
    return result


def fetch_data(path="."):
    """Sync DVC and download data from backblaze with error handling."""
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    try:
        run(f"dvc pull {path}")
        print("Data retrieval successful.")
    except subprocess.CalledProcessError as e:
        print("Error during DVC data retrieval.")
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
        run("apt-get install -y pigz pv")
        run("pigz -d -p 4 /content/my_env.tar.gz")  # Decompress using 4 cores (adjust as needed)
        print("Untarring my_env.tar.gz")
        run("pv /content/my_env.tar | tar -xf - -C /content")


def upload_my_env(upload_env=False):
    if (not upload_env):
        return
    b2_rw = get_b2_resource(os.getenv("BB_ENDPOINT"), os.getenv("BB_ACCESS_KEY_RW_ID"), userdata.get("B2_APP_KEY_RW"))
    print("my_env.tar not found, proceeding with compression.")
    # Change directory (if needed) to ensure paths are correct
    os.chdir("/content")
    # Compress the tar archive with pigz, showing a progress indicator
    # Compresses 5.5G to 3.1G
    run("apt-get install -y pigz pv")
    run("time tar cf - my_env | pv -p -e -r -b | pigz -p 4 -1 > my_env.tar.gz")
    # Upload to BackBlaze
    if userdata.get("B2_APP_KEY_RW"):
        response = upload_file(os.getenv("BB_BUCKET"), "/content", "my_env.tar.gz", b2_rw)
        print("Upload response:", response)
    else:
        print("Skipping upload to BackBlaze because no B2_APP_KEY_RW")

def create_my_env():
    workspaceDir = "/content"
    GPTNeoXDir = workspaceDir + "/GPT-NeoX"
    os.chdir(workspaceDir)
    # Check if the directory does not exist
    if not os.path.isdir(f"{workspaceDir}/my_env"):
        # Install venv package for Python 3.10
        run("apt-get update && apt-get install -y python3.10-venv")
        run("pip install virtualenv")
        # Create the virtual environment
        run("python3 -m venv {workspaceDir}/my_env")
        os.chdir(PTNeoXDir)
        # Install specific versions of torch and other packages to avoid compatibility issues
        run(f"source {workspaceDir}/my_env/bin/activate && pip install torch==2.3.0 torchaudio==2.3.0 torchvision==0.18.0 transformers==4.41.0 sentence-transformers==2.2.2")
        # Install dependencies
        run(f"source {workspaceDir}/my_env/bin/activate && pip install -r ./requirements/requirements.txt")
        run(f"source {workspaceDir}/my_env/bin/activate && pip install -r ./requirements/requirements-tensorboard.txt")


def find_project_root(current_path: Path = Path.cwd()) -> Path:
    for parent in current_path.parents:
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return current_path  # Fall back to the current path if not found