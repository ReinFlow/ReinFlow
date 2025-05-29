import os
import glob

# Define the MIT License text
LICENSE_TEXT_DPPO = """# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""


LICENSE_TEXT_FQL = """The MIT License (MIT)

# Copyright (c) 2025 FQL Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""


LICENSE_TEXT=LICENSE_TEXT_DPPO

# Define the directories to process
DIRECTORIES = [
    "env"
    "script/dataset"
    "model/rl",
    "model/gaussian",
    "model/diffusion",
    "model/common",
    "agent/finetune/diffusion_baselines",
    "agent/finetune/dppo",
]

def add_license_to_file(filepath):
    """
    Adds the MIT License to the specified Python file if it doesn't already contain it.
    Preserves the existing content of the file.
    """
    try:
        # Read the existing content
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        # Check if the license is already present
        if "MIT License" in content and "Copyright (c) 2024 Intelligent Robot Motion Lab" in content:
            print(f"License already present in {filepath}, skipping.")
            return

        # Write the license followed by the original content
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(LICENSE_TEXT + "\n" + content)
            print(f"Added license to {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def main():
    """
    Iterates through all Python files in the specified directories and adds the MIT License.
    """
    for directory in DIRECTORIES:
        # Ensure the directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, skipping.")
            continue

        # Find all Python files in the directory
        python_files = glob.glob(os.path.join(directory, "**", "*.py"), recursive=True)

        if not python_files:
            print(f"No Python files found in {directory}.")
            continue

        # Process each Python file
        for filepath in python_files:
            add_license_to_file(filepath)

if __name__ == "__main__":
    main()