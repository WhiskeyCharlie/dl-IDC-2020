#!/usr/bin/env python3

import csv
import sys


LABEL_INDEX = 2


def filter_csv_for_label(csv_path: str, label='/m/01g317'):
    people_rows = []
    with open(csv_path) as file:
        file_reader = csv.reader(file)
        for idx, row in enumerate(file_reader):
            if row[LABEL_INDEX] == label or idx == 0:
                people_rows.append(', '.join(row))
    return people_rows


def main():
    if len(sys.argv) <= 1:
        print('Incorrect usage', file=sys.stderr)
        exit(1)
    inp_file = sys.argv[1]
    to_download = filter_csv_for_label(inp_file)
    print('\n'.join(to_download))


if __name__ == '__main__':
    main()
