import sys
import os
import json
import argparse
from bwt import bwt_encode, text_markers_present
from lz_77 import lz77_encode


def prepare(input_file, output_file, encode, filter=None, limit=100000):
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

    skipped = max(len(data) - limit, 0)
    data = data[-limit:]
    new_data = []
    for item in data:
        if not isinstance(item, dict) or 'instruction' not in item or 'input' not in item or 'output' not in item:
            print(f"Error: Input file '{input_file}' has incorrect structure")
            sys.exit(1)
        instr = item['instruction']
        inp = item['input']
        if filter and (filter(instr) or filter(inp)):
            skipped += 1
            continue
        instr = encode(instr)
        inp = encode(inp)
        if len(instr) > 510 or len(inp) > 510 or len(item['output']) > 510:
            skipped += 1
            continue
        new_data.append({
            'instruction': instr,
            'input': inp,
            'output': item['output']
        })

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"Written file to '{output_file}', {skipped} entries were skipped")


def choose_encode_and_filter(mode):
    if mode == "BWT":
        return bwt_encode, text_markers_present
    if mode == "LZ77":
        return lz77_encode, None
    raise RuntimeError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode JSON data.')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('--method', type=str, default='BWT', choices=['BWT', 'LZ77'], help='Encoding method to use')
    parser.add_argument('--limit', type=int, default=100000, help='Choose number of entries to pick from the dataset')

    args = parser.parse_args()

    encode, encode_filter = choose_encode_and_filter(args.method)

    prepare(args.input_file, args.output_file, encode, encode_filter, args.limit)
