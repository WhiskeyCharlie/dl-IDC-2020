import sys
import time
from typing import Union

import cv2
import numpy as np
import pyfakewebcam
import torch
from PIL import Image
from torchvision import transforms

from model import UNET
from training import IMAGE_SIZE
from utils import clean_up_mask

WEBCAM_OUTPUT_RESOLUTION = (640, 480)
BACKGROUND_IMAGE = './images/background_nature.jpeg'
MODEL_PATH = './saved_models/unet_128x_100e_2021-02-17-17-01.pt'
DESIRED_FPS = 30


def _normalize_image_to_tensor(img: Image):
    resized_img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return transforms.PILToTensor()(resized_img).float() / 255


def remove_background_single_image(src_image: Union[str, Image.Image], model: UNET, background_image=None) -> Image:
    if type(src_image) == str:
        img = Image.open(src_image)
    else:
        img = src_image
    if background_image is None:
        background = Image.new('RGBA', img.size, (0, 255, 0))
    else:
        background = Image.open(background_image)
        background = background.resize(img.size)
    img_tensor = _normalize_image_to_tensor(img).unsqueeze(0)
    mask = torch.gt(model(img_tensor).squeeze(0), 0.25)
    mask_img = transforms.ToPILImage()(mask.int() * 255).convert('L').resize(img.size, resample=Image.BICUBIC)
    img.convert('RGBA')
    result = Image.composite(img, background, mask_img)
    background.paste(img)
    return result.convert('RGB')


def main():
    model = UNET(3, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    cam = cv2.VideoCapture(0)
    out_cam = pyfakewebcam.FakeWebcam('/dev/video2', *WEBCAM_OUTPUT_RESOLUTION)
    while True:
        ret, frame = cam.read()
        if not ret:
            print('Failed to read frame from actual webcam', file=sys.stderr)
            continue
        # Open CV used BGR instead of RGB, so we correct that with this line (otherwise I look blue)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = remove_background_single_image(Image.fromarray(frame), model, background_image=BACKGROUND_IMAGE)
        out_cam.schedule_frame(np.asarray(result))
        time.sleep(1 / DESIRED_FPS)
    del cam


if __name__ == '__main__':
    main()
