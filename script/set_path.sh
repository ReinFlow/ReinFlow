#!/bin/bash

##################### Paths #####################
#### ReinFlow Authors revised from DPPO's original repository

# Set default paths
DEFAULT_DIR="${PWD}"
DEFAULT_DATA_DIR="${PWD}/data"
DEFAULT_LOG_DIR="${PWD}/log"

# Prompt the user for input, allowing overrides
read -p "Enter the place where your reinflow script lies: [default: ${DEFAULT_DIR}], press ENTER to use default: " DIR
REINFLOW_DIR=${DIR:-$DEFAULT_DIR}  # Use user input or default if input is empty

read -p "Enter the desired data directory [default: ${DEFAULT_DATA_DIR}], press ENTER to use default: " DATA_DIR
REINFLOW_DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}  # Use user input or default if input is empty

read -p "Enter the desired logging directory [default: ${DEFAULT_LOG_DIR}], press ENTER to use default: " LOG_DIR
REINFLOW_LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_DIR}  # Use user input or default if input is empty

# Export to current session
export REINFLOW_DIR="$REINFLOW_DIR"
export REINFLOW_DATA_DIR="$REINFLOW_DATA_DIR"
export REINFLOW_LOG_DIR="$REINFLOW_LOG_DIR"

# Confirm the paths with the user
echo "Script directory set to: $REINFLOW_DIR"
echo "Data directory set to: $REINFLOW_DATA_DIR"
echo "Log directory set to: $REINFLOW_LOG_DIR"

# Append environment variables to .bashrc
echo "export REINFLOW_DIR=\"$REINFLOW_DIR\"" >> ~/.bashrc
echo "export REINFLOW_DATA_DIR=\"$REINFLOW_DATA_DIR\"" >> ~/.bashrc
echo "export REINFLOW_LOG_DIR=\"$REINFLOW_LOG_DIR\"" >> ~/.bashrc

echo "Environment variables REINFLOW_DIR, REINFLOW_DATA_DIR and REINFLOW_LOG_DIR added to .bashrc and applied to the current session."

# Set verbose logging
echo -e "# verbose debug
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1">> ~/.bashrc
echo "Suppressed D4RL import errors, turned on verbose debugging for HYDRA, CUDA, and TORCH_USE_CUDA_DSA"

##################### WandB #####################

# Prompt the user for input, allowing overrides
read -p "Enter your WandB entity (username or team name), press ENTER to skip: " ENTITY

# Check if ENTITY is not empty
if [ -n "$ENTITY" ]; then
  # If ENTITY is not empty, set the environment variable
  export REINFLOW_WANDB_ENTITY="$ENTITY"

  # Confirm the entity with the user
  echo "WandB entity set to: $REINFLOW_WANDB_ENTITY"

  # Append environment variable to .bashrc
  echo "export REINFLOW_WANDB_ENTITY=\"$ENTITY\"" >> ~/.bashrc
  
  echo "Environment variable REINFLOW_WANDB_ENTITY added to .bashrc and applied to the current session."
else
  # If ENTITY is empty, skip setting the environment variable
  echo "No WandB entity provided. Please set wandb=null when running scripts to disable wandb logging and avoid error."
fi