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


"""
Test if two .npz files are the same.
"""



import numpy as np
import os

def compare_npz_files(file1_path, file2_path):
    # Check if files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("One or both files do not exist.")
        return False

    try:
        # Load both .npz files
        data1 = np.load(file1_path, allow_pickle=True)
        data2 = np.load(file2_path, allow_pickle=True)

        # Get keys from both files
        keys1 = sorted(data1.files)
        keys2 = sorted(data2.files)

        # Check if keys are identical
        if keys1 != keys2:
            print("Files have different keys:")
            print(f"File 1 keys: {keys1}")
            print(f"File 2 keys: {keys2}")
            return False

        # Compare each array
        for key in keys1:
            arr1 = data1[key]
            arr2 = data2[key]

            # Check if arrays are equal
            if not np.array_equal(arr1, arr2):
                print(f"Arrays for key '{key}' are different.")
                return False

            # Check if dtypes are the same
            if arr1.dtype != arr2.dtype:
                print(f"Dtypes for key '{key}' are different: {arr1.dtype} vs {arr2.dtype}")
                return False

        print("Files are identical.")
        return True

    except Exception as e:
        print(f"Error comparing files: {str(e)}")
        return False

# File paths
file1_path = "data-offline/gym/ant-medium-expert-v0/train.npz"
file2_path = "data/gym/ant-medium-expert-v0/train.npz"

# Run comparison
compare_npz_files(file1_path, file2_path)