import sys

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
