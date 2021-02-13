#!/usr/bin/env python3

import csv
import pathlib
import shutil
import sys


LABEL_INDEX = 2
MASK_PATH_INDEX = 0


def copy_people_images(csv_path: str, src_dir: str, target_dir: str):
    with open(csv_path) as file:
        file_reader = csv.reader(file)
        for idx, row in enumerate(file_reader):
            if idx == 0:
                continue
            mask_path = row[MASK_PATH_INDEX]
            copy_file_if_exists(mask_path, src_dir, target_dir)


# noinspection PyBroadException
def copy_file_if_exists(file_name, src_dir, target_dir):
    file_path = pathlib.Path(src_dir) / file_name
    try:
        shutil.copy2(file_path, target_dir)
    except Exception:
        pass


def main():
    if len(sys.argv) != 4:
        print('Incorrect usage', file=sys.stderr)
        exit(1)
    inp_csv = sys.argv[1]
    inp_dir = sys.argv[2]
    output_dir = sys.argv[3]
    copy_people_images(inp_csv, inp_dir, output_dir)


if __name__ == '__main__':
    main()
