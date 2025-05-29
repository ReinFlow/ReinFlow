import os 
os.makedirs('./util/merge_wandb_tmpfiles/', exist_ok=True)

import pandas as pd
import numpy as np
import wandb
# # Define configuration for the new run
# ENV_NAME = 'robomimic'
# TASK_NAME = 'square'
# SEED = 0
# DATE_STAMP = '2025-05-13_14-00-20_2'
# MODEL_NAME = 'reflow'
# WANDB_ENTITY = 
# WANDB_PROJECT = f"{ENV_NAME}-{TASK_NAME}-finetune"
# WANDB_RUN_ID = f"{DATE_STAMP}_square_ft_reflow_logitnormal_mlp_img_td4_td1_seed0" #f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_ta4_td4_seed{SEED}_merged1"
# WANDB_RUN_NAME = f"{DATE_STAMP}_square_ft_reflow_logitnormal_mlp_img_td4_td1_seed0" #f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_ta4_td4_seed{SEED}_merged1"

# # Define the run IDs of the two online runs to merge
# run_id1 = "runs/qgua58ha"
# run_id2 = "runs/jt8m0m5o"


import os 
os.makedirs('./util/merge_wandb_tmpfiles/', exist_ok=True)

import pandas as pd
import numpy as np
import wandb
# Define configuration for the new run
ENV_NAME = 'gym'
TASK_NAME = 'walker2d-medium-v2'
SEED = 0
DATE_STAMP = '2025-05-22_11-41-25'
MODEL_NAME = 'diffusion'
DENOISE_STEP=20
WANDB_ENTITY = ""
WANDB_PROJECT = f"{ENV_NAME}-{TASK_NAME}-finetune"
WANDB_RUN_ID = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_ta4_td{DENOISE_STEP}_seed{SEED}_merged2"
WANDB_RUN_NAME = f"{DATE_STAMP}_{TASK_NAME}_ppo_{MODEL_NAME}_mlp_ta4_td{DENOISE_STEP}_seed{SEED}_merged2"

# Define the run IDs of the two online runs to merge
run_id1 = "runs/b51z9pe5"
run_id2 = "runs/q28z7fkz"




# Construct the full run paths for online access
run_path1 = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id1}"
run_path2 = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id2}"

def extract_metrics_from_online_run(run_path):
    """Extract metrics from an online W&B run using wandb.Api."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load run {run_path}: {str(e)}")
    
    try:
        history = run.history()
    except Exception as e:
        raise RuntimeError(f"Failed to extract history from {run_path}: {str(e)}")
    
    if history.empty:
        raise ValueError(f"No metrics found in run {run_path}.")
    
    if '_step' not in history.columns:
        history['_step'] = np.arange(len(history))
    
    print(f"Extracted {len(history)} rows from {run_path} with columns: {list(history.columns)}")
    return history

# Extract metrics from online runs
try:
    history1 = extract_metrics_from_online_run(run_path1)
    history2 = extract_metrics_from_online_run(run_path2)
except Exception as e:
    print(f"Error extracting metrics: {str(e)}")
    exit(1)

# Save to CSV for inspection
history1.to_csv("./util/merge_wandb_tmpfiles/run1_history.csv", index=False)
history2.to_csv("./util/merge_wandb_tmpfiles/run2_history.csv", index=False)

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

# Save merged data
combined_history.to_csv("./util/merge_wandb_tmpfiles/combined_history.csv", index=False)
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

# Finish the run
wandb.finish()

print(f"Successfully concatenated runs and logged to new run: {WANDB_RUN_ID}")
print(f"Check the new run at: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{WANDB_RUN_ID}")