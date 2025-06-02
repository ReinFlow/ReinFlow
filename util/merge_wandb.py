# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import pandas as pd
import numpy as np
import wandb

# Define configuration for the new run
ENV_NAME='robomimic'
TASK_NAME = 'square'
SEED = 0

# Local run directories and run IDs to merge.
run_dir1, run_dir2 = ["./wandb/wandb/run-20250504_092028-uohiri0i", "ReinFlow/wandb_merge/run-20250504_063813-juy1txy0"]

DATE_STAMP = '2025-05-04_14-00-20'
MODEL_NAME = 'reflow'
WANDB_ENTITY = ""
WANDB_PROJECT = f"{ENV_NAME}-{TASK_NAME}-finetune"
WANDB_RUN_ID = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_seed{SEED}_333"
WANDB_RUN_NAME = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_ta4_td4_seed{SEED}_333"
run_id1 = run_dir1.split('-')[-1]
run_id2 = run_dir2.split('-')[-1]

def extract_metrics_from_wandb_file(run_dir, run_id):
    """Extract metrics from W&B run's .wandb file using wandb.Api."""
    # Find the .wandb file in the run directory
    wandb_files = [f for f in os.listdir(run_dir) if f.endswith(".wandb")]
    if not wandb_files:
        raise FileNotFoundError(f"No .wandb files found in {run_dir}")
    
    # Use the first .wandb file
    wandb_file = wandb_files[0]
    print(f"Processing {wandb_file} with run ID: {run_id}")
    
    # Set up W&B environment
    os.environ["WANDB_DIR"] = run_dir
    os.environ["WANDB_PROJECT"] = f"{ENV_NAME}-{TASK_NAME}-finetune"
    
    # Use wandb.Api to access local run
    api = wandb.Api()
    run_path = f"""{os.environ["WANDB_ENTITY"]}/{ENV_NAME}-{TASK_NAME}-finetune/{run_id}"""
    try:
        run = api.run(run_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load run {run_path} from {run_dir}: {str(e)}")
    
    # Extract history as a DataFrame
    try:
        history = run.history()
    except Exception as e:
        raise RuntimeError(f"Failed to extract history from {run_path}: {str(e)}")
    
    if history.empty:
        raise ValueError(f"No metrics found in {wandb_file}. Check if metrics were logged correctly.")
    
    # Ensure '_step' column exists
    if '_step' not in history.columns:
        history['_step'] = np.arange(len(history))
    
    print(f"Extracted {len(history)} rows with columns: {list(history.columns)}")
    return history


# Verify directories exist
if not os.path.exists(run_dir1) or not os.path.exists(run_dir2):
    raise FileNotFoundError("One or both run directories do not exist")

# Extract metrics from .wandb files
try:
    history1 = extract_metrics_from_wandb_file(run_dir1, run_id1)
    history2 = extract_metrics_from_wandb_file(run_dir2, run_id2)
except Exception as e:
    print(f"Error extracting metrics: {str(e)}")
    exit(1)

# Save to CSV for inspection
history1.to_csv("run1_history.csv", index=False)
history2.to_csv("run2_history.csv", index=False)

# Automatic step adjustment
if '_step' in history1.columns and '_step' in history2.columns:
    max_step1 = history1['_step'].max()
    history2['_step'] = history2['_step'] + max_step1 + 1
    print(f"Adjusted steps: history1 max step = {max_step1}, history2 steps start at {max_step1 + 1}")
elif '_step' not in history1.columns and '_step' not in history2.columns:
    # If no '_step' column, create one based on row index
    history1['_step'] = np.arange(len(history1))
    history2['_step'] = np.arange(len(history2)) + len(history1)
    print("Created synthetic '_step' columns based on row indices")
else:
    raise ValueError("Inconsistent '_step' columns between runs. Please check the data.")

# Concatenate the histories
combined_history = pd.concat([history1, history2], ignore_index=True)

# Sort by '_step' to ensure chronological order
combined_history = combined_history.sort_values('_step').reset_index(drop=True)

# # Truncate to steps below 301
# combined_history = combined_history[combined_history['_step'] < 350]
# print(f"Truncated combined history to steps below 350: {len(combined_history)} rows with columns: {list(combined_history.columns)}")

# Save merged data
combined_history.to_csv("combined_history.csv", index=False)
print(f"Combined history: {len(combined_history)} rows with columns: {list(combined_history.columns)}")

# Initialize a new W&B run
try:
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=WANDB_RUN_ID,
        name=WANDB_RUN_NAME,
        config={"task_name": TASK_NAME, "seed": SEED, "model_name": MODEL_NAME},
        reinit=True
    )
except Exception as e:
    print(f"Error initializing new W&B run: {str(e)}")
    exit(1)

# Log the combined history
for _, row in combined_history.iterrows():
    # Convert row to dictionary and remove NaN values
    log_dict = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
    wandb.log(log_dict)

# Optionally log artifacts from the original runs (e.g., model checkpoints)
for run_dir in [run_dir1, run_dir2]:
    files_dir = os.path.join(run_dir, "files")
    if os.path.exists(files_dir):
        for file_name in os.listdir(files_dir):
            file_path = os.path.join(files_dir, file_name)
            if os.path.isfile(file_path):
                try:
                    artifact = wandb.Artifact(name=os.path.basename(file_path), type="model")
                    artifact.add_file(file_path)
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Warning: Failed to log artifact {file_path}: {str(e)}")

# Finish the run
wandb.finish()

print(f"Successfully concatenated runs and logged to new run: {WANDB_RUN_ID}")
print(f"Check the new run at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{WANDB_RUN_ID}")
