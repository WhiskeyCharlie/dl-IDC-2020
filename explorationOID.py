import pandas as pd
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw
import os


VALIDATION_BBOX_PATH = './oid/validation-annotations-bbox.csv'
TESTING_BBOX_PATH = './oid/test-annotations-bbox.csv'

VALIDATION_DIRECTORY_PATH = './oid/validation'
TESTING_DIRECTORY_PATH = './oid/test'

VALIDATION_ANNOTATED_DIRECTORY_PATH = './oid/validation_annotated/'
# TESTING_ANNOTATED_DIRECTORY_PATH = './oid/test_annotated'

VALIDATION_DIRECTORY_GLOB_PATH = './oid/validation/*'
TESTING_DIRECTORY_GLOB_PATH = './oid/test/*'

KEEP_COLS = ['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax']


def filter_df_people_only(df: pd.DataFrame, people_label='/m/01g317'):
    return df[df['LabelName'] == people_label]


def drop_irrelevant_cols(df: pd.DataFrame):
    return df[KEEP_COLS]


def drop_rows_of_non_existent_images(df: pd.DataFrame):
    all_images = set(map(os.path.basename, glob(VALIDATION_DIRECTORY_GLOB_PATH, recursive=False)))
    return df[~df['ImageID'].isin(all_images)]


def annotate_images(src_directory_glob: str, target_directory: str, annotation_df: pd.DataFrame):
    # Note: we only take the first annotation for an image, therefore the number of annotated images will be exactly
    # equal to the number of images in the scr_directory.
    annotated_images = set()
    all_src_images = glob(src_directory_glob, recursive=False)
    for image in all_src_images:
        filename = Path(image).stem
        if filename in annotated_images:
            continue

        # annotate the image
        first_annotation_row = annotation_df.loc[annotation_df['ImageID'] == filename].iloc[0]
        x_min, y_min = first_annotation_row.XMin, first_annotation_row.YMin
        x_max, y_max = first_annotation_row.XMax, first_annotation_row.YMax

        base_image = Image.open(f'{image}').convert('RGBA')
        width, height = base_image.size

        x1 = int(x_min * width)
        y1 = int(y_min * height)

        x2 = int(x_max * width)
        y2 = int(y_max * height)

        new_image = Image.new('RGBA', base_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(new_image)
        draw.rectangle(xy=[(x1, y1), (x2, y2)], width=3, outline=(153, 50, 204, 192))

        out = Image.alpha_composite(base_image, new_image).convert('RGB')
        out.save(f'{target_directory}{filename}.jpg', 'JPEG')

        annotated_images.add(filename)


def main():
    validation_df = pd.read_csv(VALIDATION_BBOX_PATH)
    validation_df = filter_df_people_only(validation_df)
    validation_df = drop_irrelevant_cols(validation_df)
    validation_df = drop_rows_of_non_existent_images(validation_df)
    annotate_images(VALIDATION_DIRECTORY_GLOB_PATH, VALIDATION_ANNOTATED_DIRECTORY_PATH, validation_df)


if __name__ == '__main__':
    main()
