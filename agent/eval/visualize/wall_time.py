

# Data organized as a dictionary
data = {
    'Hopper-v2': {
        'ReinFlow-R': {'avg': 11.715, 'std': 0.123},
        'ReinFlow-S': {'avg': 12.290, 'std': 0.141},
        'DPPO': {'avg': 99.046, 'std': 0.890},
        'FQL': {'avg': 4.418, 'std': 0.084}
    },
    'Walker-v2': {
        'ReinFlow-R': {'avg': 11.563, 'std': 0.260},
        'ReinFlow-S': {'avg': 13.019, 'std': 0.841},
        'DPPO': {'avg': 101.915, 'std': 3.884},
        'FQL': {'avg': 5.017, 'std': 0.365}
    },
    'Ant-v0': {
        'ReinFlow-R': {'avg': 17.473, 'std': 0.242},
        'ReinFlow-S': {'avg': 17.734, 'std': 0.407},
        'DPPO': {'avg': 102.012, 'std': 2.811},
        'FQL': {'avg': 5.167, 'std': 0.191}
    },
    'Humanoid-v3': {
        'ReinFlow-R': {'avg': 30.916, 'std': 0.625},
        'ReinFlow-S': {'avg': 30.529, 'std': 0.486},
        'DPPO': {'avg': 109.566, 'std': 3.961},
        'FQL': {'avg': None, 'std': None}
    },
    'Franka Kitchen-complete': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 26.537, 'std': 0.182},
        'DPPO': {'avg': 83.158, 'std': 1.533},
        'FQL': {'avg': 5.249, 'std': 0.271}
    },
    'Franka Kitchen-mixed': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 26.537, 'std': 0.182},
        'DPPO': {'avg': 83.158, 'std': 1.533},
        'FQL': {'avg': 5.249, 'std': 0.271}
    },
    'Franka Kitchen-partial': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 26.537, 'std': 0.182},
        'DPPO': {'avg': 83.158, 'std': 1.533},
        'FQL': {'avg': 5.249, 'std': 0.271}
    },
    'Can-img': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 218.061, 'std': 1.734},
        'DPPO': {'avg': 308.933, 'std': 1.771},
        'FQL': {'avg': None, 'std': None}
    },
    'Square-img': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 313.206, 'std': 0.811},
        'DPPO': {'avg': 437.830, 'std': 2.782},
        'FQL': {'avg': None, 'std': None}
    },
    'Transport-img': {
        'ReinFlow-R': {'avg': None, 'std': None},
        'ReinFlow-S': {'avg': 487.505 , 'std': 0.000},
        'DPPO': {'avg': 701.712, 'std': 17.052},
        'FQL': {'avg': None, 'std': None}
    }
}
# Task categorization
locomotion_tasks = ['Hopper-v2', 'Walker-v2', 'Ant-v0', 'Humanoid-v3']
manipulation_tasks = ['Franka Kitchen-complete', 'Franka Kitchen-mixed', 'Franka Kitchen-partial', 'Can-img', 'Square-img', 'Transport-img']

# Function to calculate percentage time saved
def calculate_time_saved(dppo_avg, reinflow_avg):
    if dppo_avg is None or reinflow_avg is None:
        return None
    return ((dppo_avg - reinflow_avg) / dppo_avg) * 100

# Function to compute average time saved for a group of tasks
def compute_average_time_saved(tasks, method):
    time_saved_values = []
    for task in tasks:
        reinflow_avg = data[task][method]['avg']
        dppo_avg = data[task]['DPPO']['avg']
        time_saved = calculate_time_saved(dppo_avg, reinflow_avg)
        if time_saved is not None:
            time_saved_values.append(time_saved)
    return sum(time_saved_values) / len(time_saved_values) if time_saved_values else None

# Compute and print average time saved for locomotion tasks
print("Average Percentage Time Saved for Locomotion Tasks:")
for method in ['ReinFlow-R', 'ReinFlow-S']:
    avg_time_saved = compute_average_time_saved(locomotion_tasks, method)
    if avg_time_saved is not None:
        print(f"  {method}: {avg_time_saved:.2f}%")
    else:
        print(f"  {method}: Not available (missing data)")

# Compute and print average time saved for manipulation tasks
print("\nAverage Percentage Time Saved for Manipulation Tasks:")
for method in ['ReinFlow-R', 'ReinFlow-S']:
    avg_time_saved = compute_average_time_saved(manipulation_tasks, method)
    if avg_time_saved is not None:
        print(f"  {method}: {avg_time_saved:.2f}%")
    else:
        print(f"  {method}: Not available (missing data)")