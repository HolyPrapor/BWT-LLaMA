import sys
import os
import json


def BWT_encode(s):
    """Apply Burrows-Wheeler transform to input string."""
    assert "\002" not in s and "\003" not in s, "Input string cannot contain STX and ETX characters"
    s = "\002" + s + "\003"  # Add start and end of text marker
    table = sorted(s[i:] + s[:i] for i in range(len(s)))  # Table of rotations of string
    last_column = [row[-1:] for row in table]  # Last characters of each row
    return "".join(last_column)  # Convert list of characters into string


def BWT_decode(s):
    """Decode Burrows-Wheeler transform of input string."""
    table = [""] * len(s)  # Initialize table
    for _ in s:
        table = sorted(s[i] + table[i] for i in range(len(s)))  # Insert characters of s
    # Find row that ends with ETX character
    for row in table:
        if row.endswith("\003"):
            return row.rstrip("\003").lstrip("\002")  # Strip STX and ETX characters


def main(input_file, output_file):
    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Check if output path is valid
    if not os.path.isdir(os.path.dirname(output_file)):
        print(f"Error: Output path '{os.path.dirname(output_file)}' does not exist.")
        sys.exit(1)

    with open(input_file, 'r') as f:
        data = json.load(f)

    # encode 'instruction' and 'input' fields
    for item in data:
        item['instruction'] = BWT_encode(item['instruction'])
        item['input'] = BWT_encode(item['input'])

    with open(output_file, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <input_file> <output_file>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
