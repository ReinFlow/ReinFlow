"""
This file reads in any .hdf5 file and shows its dictionary structure.
"""

import h5py

# Open the HDF5 file
file_path =  "hopper_medium-v2.hdf5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

with h5py.File(file_path, 'r') as f:
    # Print the structure of the file
    print("File Structure:")
    f.visititems(print_structure)

    # If you want to see the keys at the root level
    print("\nRoot level keys:")
    print(list(f.keys()))
    
    # If you want to see a sample of data (for the first dataset found)
    for name, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            print(f"\nSample data from {name}:")
            print(type(obj[0]), obj[0].shape)  # Print first 5 elements
            break



'''
Dataset: actions
  Shape: (2000000, 8)
  Type: float32
Dataset: observations
  Shape: (2000000, 111)
  Type: float32
Dataset: rewards
  Shape: (2000000,)
  Type: float32
Dataset: terminals
  Shape: (2000000,)
  Type: bool
Dataset: timeouts
  Shape: (2000000,)
  Type: bool

Root level keys:
['actions', 'observations', 'rewards', 'terminals', 'timeouts']
'''

