import os
import wandb
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def extract_metrics_from_event_files(run_dir):
    """Extract scalar metrics from TensorBoard event files in a W&B run directory."""
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    # Get all scalar tags (metric names)
    scalar_tags = ea.Tags()['scalars']
    
    # Initialize dictionary to store metrics
    metrics = {'_step': []}
    for tag in scalar_tags:
        metrics[tag] = []
    
    # Extract scalar values and steps
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        for event in events:
            if tag not in metrics:
                metrics[tag] = [np.nan] * len(metrics['_step'])
            while len(metrics[tag]) < len(metrics['_step']):
                metrics[tag].append(np.nan)
            metrics[tag].append(event.value)
        if tag == scalar_tags[0]:
            metrics['_step'] = [event.step for event in events]
    
    # Ensure all lists are the same length
    max_len = len(metrics['_step'])
    for tag in metrics:
        metrics[tag] += [np.nan] * (max_len - len(metrics[tag]))
    
    return pd.DataFrame(metrics)

# Define local run directories
run_dir1 = "./wandb/wandb/run-20250430_154201-n78ls5g2"
run_dir2 = "./wandb/wandb/run-20250501_125851-phs3soka"

# Extract metrics from local event files
history1 = extract_metrics_from_event_files(run_dir1)
history2 = extract_metrics_from_event_files(run_dir2)

# Save to CSV for inspection (optional)
history1.to_csv("run1_history.csv", index=False)
history2.to_csv("run2_history.csv", index=False)

# Automatic step adjustment
if '_step' in history1.columns and '_step' in history2.columns:
    print(f" '_step' column exists. Automatically concatenate. ")
    max_step1 = history1['_step'].max()
    history2['_step'] = history2['_step'] + max_step1 + 1
elif '_step' not in history1.columns and '_step' not in history2.columns:
    print(f"If no '_step' column, create one based on row index")
    history1['_step'] = np.arange(len(history1))
    history2['_step'] = np.arange(len(history2)) + len(history1)
else:
    raise ValueError("Inconsistent '_step' columns between runs. Please check the data.")

# Concatenate the histories
combined_history = pd.concat([history1, history2], ignore_index=True)

# Sort by '_step' to ensure chronological order
combined_history = combined_history.sort_values('_step').reset_index(drop=True)

# Save merged data (optional)
combined_history.to_csv("combined_history.csv", index=False)

# Define configuration for the new run
TASK_NAME = 'hopper-medium-v2'
SEED = 42
WANDB_PROJECT = f"gym-hopper-medium-v2-finetune"
DATE_STAMP = '2025-05-01_13-11-02'
MODEL_NAME = 'shortcut'
WANDB_ENTITY = ""
WANDB_RUN_ID = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_td4_td4_seed{SEED}"
WANDB_RUN_NAME = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_td4_td4_seed{SEED}"

# Initialize a new W&B run
wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    id=WANDB_RUN_ID,
    name=WANDB_RUN_NAME,
    config={"task_name": TASK_NAME, "seed": SEED, "model_name": MODEL_NAME},
    reinit=True
)

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
                artifact = wandb.Artifact(name=os.path.basename(file_path), type="model")
                artifact.add_file(file_path)
                wandb.log_artifact(artifact)

# Finish the run
wandb.finish()

print(f"Successfully concatenated runs and logged to new run: {WANDB_RUN_ID}")
print(f"Check the new run at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{WANDB_RUN_ID}")