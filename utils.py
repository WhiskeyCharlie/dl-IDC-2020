import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def clean_up_mask(mask: Image):
    cv_mask = np.array(mask)
    kernel = np.ones((13, 13), np.uint8)  # hardcoded 13 simply gave nice results
    opened_mask = cv.morphologyEx(cv_mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the person component (biggest component after background)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)

    if num_labels > 1:
        # step2.1: find the background component
        h, _ = labels.shape  # get mask height
        discriminant_subspace = labels[:int(h/10), :]
        bkg_index = np.argmax(np.bincount(discriminant_subspace.flatten()))

        # step2.2: biggest component after background is person (that's a highly probable hypothesis)
        blob_areas = []
        for i in range(0, num_labels):
            blob_areas.append(stats[i, cv.CC_STAT_AREA])
        blob_areas = list(zip(range(len(blob_areas)), blob_areas))
        blob_areas.sort(key=lambda tup: tup[1], reverse=True)  # sort from biggest to smallest area components
        blob_areas = [a for a in blob_areas if a[0] != bkg_index]  # remove background component
        person_index = blob_areas[0][0]  # biggest component that is not background is presumably person
        processed_mask = np.uint8((labels == person_index) * 255)

        return processed_mask
    else:  # only 1 component found (probably background) we don't need further processing
        return opened_mask


def _cv_image_to_pil(image) -> Image:
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def normalize_image_to_tensor(img: Image, output_size):
    resized_img = img.resize((output_size, output_size))
    return transforms.PILToTensor()(resized_img).float() / 255.0
