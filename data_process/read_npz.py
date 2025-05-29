"""
This file reads in any .npz file and shows its dictionary structure.
"""



import numpy as np
import argparse
import os
from zipfile import ZipFile, BadZipFile

def main():
    # Setting up argument parser
    parser = argparse.ArgumentParser(description="Inspect NPZ file contents")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the NPZ file')
    parser.add_argument('--print_data_sample', action='store_true', help='Print sample data from arrays')
    args = parser.parse_args()
    
    file_path = args.data_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
        
    # Check if file is empty or too small to be a valid NPZ
    if os.path.getsize(file_path) < 100:  # NPZ files typically have a header, so <100 bytes is likely invalid
        print(f"Error: File {file_path} is empty or too small to be a valid NPZ file (size: {os.path.getsize(file_path)} bytes)")
        return
    
    # Check if file is a valid ZIP (NPZ is a ZIP archive)
    try:
        with ZipFile(file_path, 'r') as zip_check:
            zip_check.testzip()  # Tests integrity of the ZIP file
    except BadZipFile:
        print(f"Error: File {file_path} is not a valid NPZ file (corrupted or invalid ZIP archive)")
        return
    except Exception as e:
        print(f"Error: File {file_path} cannot be processed as an NPZ file ({str(e)})")
        return
    
    try:
        # Load the .npz file
        with np.load(file_path, allow_pickle=False) as data:
            # Check if the NPZ file contains any arrays
            if not data.files:
                print(f"Error: File {file_path} contains no arrays (empty NPZ file)")
                return

            # Print the keys (names of the arrays stored in the file)
            print(f"Arrays in the {file_path} file:")
            print(data.files)

            # Print information about each array
            for key in data.files:
                array = data[key]
                array: np.ndarray
                print(f"\nArray: {key}")
                print(f"Shape: {array.shape}")
                print(f"Data type: {array.dtype}")
                
                # Check if array is empty before computing min/max
                if array.size == 0:
                    print("Warning: Array is empty (zero size)")
                    continue
                
                try:
                    print(f"Data min: {array.min(axis=0)}")
                    print(f"Data max: {array.max(axis=0)}")
                except ValueError as ve:
                    print(f"Warning: Cannot compute min/max for array '{key}' ({str(ve)})")
                
                # Print a small sample of the data (first 5 elements or first 5 rows) if enabled
                if args.print_data_sample:
                    if array.size == 0:
                        print("Sample data: (empty array)")
                    else:
                        if array.ndim == 1:
                            print("Sample data (first 5 elements):")
                            print(array[:5])
                        else:
                            print("Sample data (first 5 rows):")
                            print(array[:5])
                        
                        # If the array is too large, you might want to limit the output
                        if array.size > 1000:
                            print("(Output truncated due to large array size)")
                
    except ValueError as ve:
        print(f"Error loading {file_path}: Invalid NPZ file format ({str(ve)})")
    except Exception as e:
        print(f"Error loading or processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main()