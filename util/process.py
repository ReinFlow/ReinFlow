
import numpy as np
from typing import List

# def set_axis_in_scientific(ax, xyz:str, scientific_fontsize):
#     """
#     # Set x,y,or z axis ticks to scientific notation
#     """
#     import matplotlib.ticker as ticker
#     if xyz=='x':
#         xyzaxis = ax.xaxis
#     elif xyz=='y':
#         xyzaxis = ax.yaxis
#     elif xyz=='z':
#         xyzaxis = ax.zaxis
#     else:
#         raise ValueError(f"You got the xyz attributed wrong. it should be within ['x','y','z'].")
#     formatter = ticker.ScalarFormatter(useMathText=True)
#     formatter.set_scientific(True)
#     # formatter.set_powerlimits((-2, 2))  # Flexible range for scientific notation
#     xyzaxis.set_major_formatter(formatter)

#     # Ensure offset text is visible
#     offset_text = xyzaxis.get_offset_text()
#     offset_text.set_fontsize(scientific_fontsize)
#     offset_text.set_visible(True)
#     # offset_text.set_position((1, 1))  # Adjust position if needed

#     # ax.ticklabel_format(style='sci', axis=xyz, scilimits=(-2, 2))
    
    
    
            

def sort_handles_by_labels(handles:list, labels:List[str], desired_order:List[str]):
    """
    Show the legends of your curves in desired order. 
    
    **Use case**
    ```python
    handle, = ax.plot(..., label=xxx, ...)
    handles.append(handle)
    labels.append(label)
    
    sorted_handles, sorted_labels = sort_handles_by_labels(handles, labels, desired_order)
    
    ax.legend(sorted_handles, sorted_labels)
    ```
    """
    sorted_handles_labels=sorted(zip(handles, labels), key=lambda x: desired_order.index(x[1]) if x[1] in desired_order else len(desired_order))
    sorted_handles, sorted_labels=zip(*sorted_handles_labels)
    return  sorted_handles, sorted_labels


def find_duplicates_in_tensor(tensor):
    # Convert tensor to numpy array
    arr = tensor.numpy().flatten()
    
    # Find unique elements and their counts
    unique, counts = np.unique(arr, return_counts=True)
    
    # Create a dictionary of element:count pairs
    element_counts = dict(zip(unique, counts))
    
    # Find duplicates (elements with count > 1)
    duplicates = {elem: count for elem, count in element_counts.items() if count > 1}
    
    # Create a list of duplicates with their IDs and frequencies
    result = []
    for elem, count in duplicates.items():
        indices = np.where(arr == elem)[0]
        result.append({
            'value': elem,
            'frequency': count,
            'ids': indices.tolist()
        })
    
    return result


def is_multiple_of_one_over_k(value: float, K: int) -> bool:
    # Multiply the value by K and check if it's close to an integer
    product = value * K
    # Use a small epsilon to account for floating-point imprecision
    epsilon = 1e-10
    return abs(product - round(product)) < epsilon



