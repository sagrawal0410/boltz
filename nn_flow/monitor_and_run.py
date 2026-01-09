import os
import subprocess
import time
from datetime import datetime
import wandb

SLEEP_INTERVAL = 60  # seconds

def run_command(command):
    """Runs a command and prints its output."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result

def main():
    """Monitors for git updates and launches a job."""
    wandb.init(
        project="nn_flow_monitor",
        name=f"monitor_and_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    
    print("Starting monitoring script.")
    
    while True:
        print("\n" + "="*80)
        print(f"Timestamp: {datetime.now()}")
        
        print("Running git pull...")
        pull_result = run_command(["git", "pull"])

        up_to_date_messages = ["Already up to date.", "Already up-to-date."]

        if pull_result.returncode != 0:
            print("Git pull failed. Skipping job run.")
        elif not any(msg in pull_result.stdout for msg in up_to_date_messages):
            print("--- Git pull detected updates. Launching job. ---")

            # Create a clean environment for the subprocess to avoid wandb conflicts.
            child_env = os.environ.copy()
            for key in list(child_env.keys()):
                if key.startswith("WANDB_"):
                    del child_env[key]

            # Using Popen to stream output in real-time
            process = subprocess.Popen(
                ["bash", "command.sh"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            
            # Log the output from the script
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")
            
            return_code = process.wait()
            print(f"--- Job finished with exit code {return_code} ---")
        else:
            print("No new updates detected from git pull.")

        print(f"Waiting for {SLEEP_INTERVAL} seconds...")
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main() 