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