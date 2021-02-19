import sys
import time
import warnings
from typing import Union

import cv2
import numpy as np
import pyfakewebcam
import torch
from PIL import Image
from torchvision import transforms

from model import UNET
from training import IMAGE_SIZE
from utils import normalize_image_to_tensor

WEBCAM_OUTPUT_RESOLUTION = (640, 640)
BACKGROUND_IMAGE = './backgrounds/default_background.jpg'
FAKE_WEBCAM_PATH = '/dev/video2'
DESIRED_FPS = 30
MODEL_PATH = './saved_models/default_model.pt'


def remove_background_single_image(src_image: Union[str, Image.Image], model: UNET, background) -> Image:
    if type(src_image) == str:
        img = Image.open(src_image)
    else:
        img = src_image
    img_tensor = normalize_image_to_tensor(img, IMAGE_SIZE).unsqueeze(0)
    res = model(img_tensor)
    mask = torch.gt(res.squeeze(0), 0.5)[0]
    mask = mask.int() * 255
    mask_img = transforms.ToPILImage()(mask).convert('L').resize(img.size, resample=Image.BICUBIC)
    img.convert('RGBA')
    result = Image.composite(img, background, mask_img)
    background.paste(img)
    return result.convert('RGB')


def emulate_webcam_loop(model: UNET, input_camera, output_camera):
    try:
        background_image = Image.open(BACKGROUND_IMAGE).resize(WEBCAM_OUTPUT_RESOLUTION)
        while True:
            start = time.time()
            ret, frame = input_camera.read()
            if not ret:
                print('Failed to read frame from actual webcam', file=sys.stderr)
                continue
            # Open CV uses BGR instead of RGB, so we correct that with this line (otherwise I look blue)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = remove_background_single_image(Image.fromarray(frame).resize(WEBCAM_OUTPUT_RESOLUTION),
                                                    model, background_image.copy())
            output_camera.schedule_frame(np.asarray(result))
            time.sleep(max(0, round((1 / DESIRED_FPS) - (time.time() - start))))
    except KeyboardInterrupt:
        print('Shutting down.')
    del input_camera
    exit(0)


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Loading UNET model')
        model = UNET(3, 1)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print('Capturing default webcam')
        cam = cv2.VideoCapture(0)
        print('Creating fake webcam')
        out_cam = pyfakewebcam.FakeWebcam(FAKE_WEBCAM_PATH, *WEBCAM_OUTPUT_RESOLUTION)
        emulate_webcam_loop(model, cam, out_cam)


if __name__ == '__main__':
    main()
