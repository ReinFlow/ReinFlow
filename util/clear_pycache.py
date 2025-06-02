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


#!/usr/bin/env python3

import os
import shutil
import argparse
from util.dirs import REINFLOW_DIR

def clean_pycache(directory):
    """
    Remove all __pycache__ directories and .pyc files in the given directory and its subfolders.
    
    Args:
        directory (str): The root directory to search for __pycache__ and .pyc files.
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} does not exist.")
        return 1

    print(f"Removing __pycache__ directories and .pyc files in {directory}...")

    # Counter for removed items
    removed_count = 0

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed directory: {pycache_path}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {pycache_path}: {e}")

        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                try:
                    os.remove(pyc_file)
                    print(f"Removed file: {pyc_file}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {pyc_file}: {e}")

    if removed_count == 0:
        print("No __pycache__ directories or .pyc files found.")
    else:
        print(f"Successfully removed {removed_count} items.")

    print("Cleanup complete.")
    return 0

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Remove __pycache__ directories and .pyc files from a directory.")
    parser.add_argument(
        'directory',
        nargs='?',
        default=REINFLOW_DIR,
        help='The directory to clean (default: REINFLOW_DIR)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the cleanup
    exit_code = clean_pycache(args.directory)
    return exit_code

if __name__ == '__main__':
    exit(main())