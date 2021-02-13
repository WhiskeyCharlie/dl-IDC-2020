#!/usr/bin/env python3
from glob import glob
from pathlib import Path

TRAIN_IMAGES_GLOB_PATH = './oid/train/*'
TO_DOWNLOAD_PATH = './oid/to_download_train.txt'
PREFIX = 'train/'


def main():
    downloaded_image_paths = [Path(image) for image in glob(TRAIN_IMAGES_GLOB_PATH)]
    downloaded_ids = set([f'{PREFIX}{path.stem}' for path in downloaded_image_paths])
    to_download_set = set()
    with open(TO_DOWNLOAD_PATH) as file:
        for line in file.readlines():
            image_descriptor = line.strip()
            to_download_set.add(image_descriptor)
            if image_descriptor not in downloaded_ids:
                print(image_descriptor)
    print(len(to_download_set))


if __name__ == '__main__':
    main()
