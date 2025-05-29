import shutil
def create_bordered_text(text, border_char='#'):
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Split the text into lines
    lines = text.split('\n')

    # Determine left and right columns
    left_lines = []
    right_lines = []
    for line in lines:
        if ':' in line:
            left, right = line.split(':', 1)
            left_lines.append(left.strip())
            right_lines.append(right.strip())
        else:
            left_lines.append(line.strip())
            right_lines.append('')

    # Calculate column widths
    left_width = max(len(line) for line in left_lines) + 2  # +2 for ' :'
    right_width = max(len(line) for line in right_lines) + 2  # +2 for ': '

    # Format the lines
    formatted_lines = [
        f"# {left.ljust(left_width)} {right.rjust(right_width)} #"
        for left, right in zip(left_lines, right_lines)
    ]

    # Create top and bottom borders
    border = border_char * terminal_width

    # Construct the final message
    message = (
        f"\n{border}\n"
        + '\n'.join(formatted_lines)
        + f"\n{border}"
    )

    return message

def print_type_and_shape(var_name, var):
    var_type = type(var)
    try:
        var_shape = var.shape  # This assumes var has a 'shape' attribute
    except AttributeError:
        var_shape = "No shape attribute"

    print(f"{var_name}: Type = {var_type}, Shape = {var_shape}")