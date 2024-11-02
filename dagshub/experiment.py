# experiment.py

import subprocess
import os
import hydra

@hydra.main(config_path="configs", config_name="config")
def run_multiple_experiments(_):
    # Path to train.py (absolute path to avoid issues)
    train_script_path = os.path.join(os.path.dirname(__file__), "train.py")
    
    # List of experiment configurations to run
    experiments = ["experiment1", "experiment2"]

    for experiment in experiments:
        try:
            # Run each experiment by calling train.py with the appropriate configuration
            print(f"\nRunning {experiment}...\n")
            result = subprocess.run(
                ["python", train_script_path, f"experiment={experiment}"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment {experiment} failed with error:\n{e}")

if __name__ == "__main__":
    run_multiple_experiments()
