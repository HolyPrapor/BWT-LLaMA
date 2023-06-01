import sys
import os
import json

# Start of text marker and end of text marker
SOTM = "~"
EOTM = "#"


def text_markers_present(s):
    return SOTM in s or EOTM in s


def bwt_encode(s):
    """Apply Burrows-Wheeler transform to input string."""
    if text_markers_present(s):
        print(f"Input string cannot contain SOTM ('{SOTM}') and EOTM ('{EOTM}') characters")
        sys.exit(1)
    s = SOTM + s + EOTM  # Add start and end of text marker
    table = sorted(s[i:] + s[:i] for i in range(len(s)))  # Table of rotations of string
    last_column = [row[-1:] for row in table]  # Last characters of each row
    return "".join(last_column)  # Convert list of characters into string


def main(input_file, output_file):
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    # Check if output path is valid
    if not os.path.isdir(os.path.dirname(output_file)):
        print(f"Error: Output path '{os.path.dirname(output_file)}' does not exist")
        sys.exit(1)

    with open(input_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Input file '{input_file}' is not a valid JSON file")
            sys.exit(1)

    # Check if data is a list and each item is a dictionary with correct keys
    if not isinstance(data, list):
        print(f"Error: Input file '{input_file}' does not contain a list at the top level")
        sys.exit(1)

    skipped = 0
    for item in data:
        if not isinstance(item, dict) or 'instruction' not in item or 'input' not in item or 'output' not in item:
            print(f"Error: Input file '{input_file}' has incorrect structure")
            sys.exit(1)
        instr = item['instruction']
        inp = item['input']
        if text_markers_present(instr) or text_markers_present(inp):
            skipped += 1
            continue
        item['instruction'] = bwt_encode(instr)
        item['input'] = bwt_encode(inp)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Written file to '{output_file}', {skipped} entries were skipped due to SOTM or EOTM presence")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <input_file> <output_file>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
