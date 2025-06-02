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
import glob
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Initialize YAML parser with ruamel.yaml to preserve comments and structure
yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

# Directory containing the YAML files
directory = "cfg/robomimic/eval"

# Find all .yaml files recursively in the specified directory
yaml_files = glob.glob(os.path.join(directory, "**/*.yaml"), recursive=True)

for yaml_file in yaml_files:
    print(f"Processing {yaml_file}...")
    
    try:
        # Read the YAML file
        with open(yaml_file, 'r') as f:
            data = yaml.load(f)
        
        # Ensure data is a dictionary
        if not isinstance(data, CommentedMap):
            print(f"  Skipping {yaml_file}: Not a valid YAML dictionary")
            continue
        
        # Check if normalization_path exists
        if 'normalization_path' in data:
            # Create a new CommentedMap to preserve order
            new_data = CommentedMap()
            found_normalization = False
            
            # Iterate through items to maintain order and insert load_ema after normalization_path
            for key, value in data.items():
                new_data[key] = value
                if key == 'normalization_path':
                    found_normalization = True
                    new_data['load_ema'] = True
            
            # If normalization_path was found, update the data
            if found_normalization:
                data.clear()
                data.update(new_data)
                
                # Write back to the file
                with open(yaml_file, 'w') as f:
                    yaml.dump(data, f)
                print(f"  Successfully added load_ema: True to {yaml_file}")
            else:
                print(f"  normalization_path found but could not insert load_ema in {yaml_file}")
        else:
            print(f"  Skipping {yaml_file}: normalization_path key not found")
    
    except Exception as e:
        print(f"  Error processing {yaml_file}: {str(e)}")

print("Processing complete.")