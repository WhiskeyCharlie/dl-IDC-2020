from model import UNET
from training import IMAGE_SIZE
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
import numpy as np


def _normalize_image_to_tensor(img: Image):
    resized_img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return torchvision.transforms.PILToTensor()(resized_img).float() / 255


def remove_background_single_image(src_image: str, model: UNET) -> Image:
    img = Image.open(src_image)
    img_tensor = _normalize_image_to_tensor(img).unsqueeze(0)
    mask = torch.gt(model(img_tensor).squeeze(0), 0.5)
    mask_img = torchvision.transforms.ToPILImage()(mask.int() * 255).convert('L').resize(img.size,
                                                                                         resample=Image.BICUBIC)
    background = Image.new('RGBA', img.size, 'BLACK')
    img.convert('RGBA')
    img.putalpha(mask_img, )
    print(np.asarray(img))
    background.paste(img)
    return background


def main():
    model = UNET(3, 1)
    model.load_state_dict(torch.load('./saved_models/unet.pt'))
    model.eval()
    plt.imshow(remove_background_single_image('./images/2021-02-16-085831.jpg', model))
    plt.show()


if __name__ == '__main__':
    main()
