from typing import Union

from model import UNET
from training import IMAGE_SIZE
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pyfakewebcam
import cv2
import time
import numpy as np

WEBCAM_OUTPUT_RESOLUTION = (640, 480)


def _normalize_image_to_tensor(img: Image):
    resized_img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return transforms.PILToTensor()(resized_img).float() / 255


def remove_background_single_image(src_image: Union[str, Image.Image], model: UNET) -> Image:
    if type(src_image) == str:
        img = Image.open(src_image)
    else:
        img = src_image
    img_tensor = _normalize_image_to_tensor(img).unsqueeze(0)
    mask = torch.gt(model(img_tensor).squeeze(0), 0.5)
    mask_img = transforms.ToPILImage()(mask.int() * 255).convert('L').resize(img.size, resample=Image.BICUBIC)
    background = Image.new('RGBA', img.size, (0, 255, 0))
    img.convert('RGBA')
    result = Image.composite(img, background, mask_img)
    background.paste(img)
    return result.convert('RGB')


def main():
    model = UNET(3, 1)
    model.load_state_dict(torch.load('./saved_models/unet.pt'))
    model.eval()
    cam = cv2.VideoCapture(0)
    out_cam = pyfakewebcam.FakeWebcam('/dev/video2', *WEBCAM_OUTPUT_RESOLUTION)
    while True:
        ret, frame = cam.read()
        if not ret:
            print('failure')
            continue
        result = remove_background_single_image(Image.fromarray(frame), model)
        out_cam.schedule_frame(np.asarray(result))
        time.sleep(1 / 30)
    del cam


if __name__ == '__main__':
    main()
