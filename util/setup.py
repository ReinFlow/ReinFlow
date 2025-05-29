import os
import subprocess

def setup_mujoco_environment():
    try:
        # Deactivate Conda environment twice
        subprocess.run("conda deactivate", shell=True, check=True)
        subprocess.run("conda deactivate", shell=True, check=True)
        
        # Activate the mujoco_py Conda environment
        subprocess.run("conda activate mujoco_py", shell=True, check=True)
        
        # Clear the console (equivalent to 'clear' command)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("MuJoCo environment setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while setting up the environment: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")