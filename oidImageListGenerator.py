#!/usr/bin/env python3

import csv
import sys


LABEL_INDEX = 2
IMAGE_ID_INDEX = 0


def filter_csv_for_label(csv_path: str, prefix: str, label='/m/01g317'):
    images = []
    with open(csv_path) as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            if row[LABEL_INDEX] == label:
                images.append(f'{prefix}/{row[IMAGE_ID_INDEX]}')
    return images


def main():
    if len(sys.argv) < 3:
        print('Incorrect usage', file=sys.stderr)
        exit(1)
    inp_file = sys.argv[1]
    prefix = sys.argv[2]
    to_download = filter_csv_for_label(inp_file, prefix)
    print('\n'.join(to_download))


if __name__ == '__main__':
    main()
