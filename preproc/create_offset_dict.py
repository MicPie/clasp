import argparse
import time
from datetime import datetime
import json


def get_args():
    """Get all parsed arguments."""
    parser = argparse.ArgumentParser(description="CLASP training offset dict creator")

    parser.add_argument("-i", type=str,
                        help="input: path preprocessed csv file for training")
    parser.add_argument("-o", type=str,
                        help="output: path preprocessed offset dictionary json file for training")

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    print(f"{datetime.now()} get the line count by loading {args.i} into RAM")
    with open(args.i, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    line_count = len(lines)
    del(lines)

    print(f"{datetime.now()} create offset dict")
    tp = time.time()
    offset_dict = {}
    with open(args.i, 'rb') as f:
        f.readline()  # move over header
        for line in range(line_count):
            offset = f.tell()
            offset_dict[line] = offset
            f.readline()
    print(f"{datetime.now()} offset dict creation time: {time.time() - tp:.3f} s")

    print(f"{datetime.now()} save offset dict to {args.o}")
    with open(args.o, 'w', encoding='utf-8') as f:
        json.dump(offset_dict, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
