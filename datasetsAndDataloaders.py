import glob
import os

import torch
import torch.utils.data as data
import torchvision
from PIL import Image


class DataLoaderSegmentation(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderSegmentation, self).__init__(*args, **kwargs)

    def positive_class_weight(self):
        return self.dataset.get_positive_weights()


class DatasetSegmentationGH(data.Dataset):
    def __init__(self, main_folder_path, segments_folder_path):
        super(DatasetSegmentationGH, self).__init__()

        mask_files = glob.glob(os.path.join(segments_folder_path, '*'))

        self._main_folder_path = main_folder_path
        self.total_positive_pixels = 0
        self.total_pixels = 0
        self.images = dict()

        for index, mask_path in enumerate(mask_files):
            self._read_single_image_pair(mask_path, index)

    def _read_single_image_pair(self, mask_path, index):
        base_name = os.path.basename(mask_path)
        image_path = os.path.join(self._main_folder_path, base_name.replace('.png', '.jpg'))
        image_pil = Image.open(image_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('1')
        image_1 = torchvision.transforms.PILToTensor()(image_pil)
        image_2 = torchvision.transforms.PILToTensor()(mask_pil)
        self.total_positive_pixels += torch.count_nonzero(image_2)
        self.total_pixels += torch.numel(image_1)
        self.images[index] = (image_1.float() / 255.0, image_2.float() / 255.0)

    def get_positive_weights(self):
        if self.total_pixels == 0:
            return 0
        return (self.total_pixels - self.total_positive_pixels) / self.total_positive_pixels

    def __getitem__(self, index):
        return self.images.get(index)

    def __len__(self):
        return len(self.images)


class DatasetSegmentationOI(data.Dataset):
    def __init__(self, main_folder_path, segments_folder_path, first_char_pattern):
        super(DatasetSegmentationOI, self).__init__()
        self.mask_files = glob.glob(os.path.join(segments_folder_path, '*.png'))
        self.img_files = []
        self.cache = dict()
        self.mask_files_with_corresponding_img = []
        for mask_path in self.mask_files:
            base_file_name = os.path.basename(mask_path)
            img_path = os.path.join(main_folder_path, os.path.basename(mask_path).split('_')[0] + '.jpg')
            # Second clause is so we only take some of the pics
            if os.path.exists(img_path) and base_file_name[0] in first_char_pattern:
                self.img_files.append(img_path)
                self.mask_files_with_corresponding_img.append(mask_path)
        self.mask_files = None
        self.total_positive_pixels = 0
        self.total_pixels = 0

    def get_positive_weights(self):
        if self.total_pixels == 0:
            return 0
        return self.total_positive_pixels / self.total_pixels

    def __getitem__(self, index):
        if index not in self.cache:
            img_path = self.img_files[index]
            mask_path = self.mask_files_with_corresponding_img[index]
            x_img = Image.open(img_path).convert("RGB")
            label = Image.open(mask_path).convert("1")
            background = Image.new('1', label.size)
            background.paste(label)
            image_1 = torchvision.transforms.PILToTensor()(x_img)
            image_2 = torchvision.transforms.PILToTensor()(background)
            x_img.close()
            label.close()
            background.close()
            inp, out = image_1.float() / 255.0, image_2.float() / 255.0
            self.cache[index] = inp, out
            self.total_pixels += torch.numel(out)
            self.total_positive_pixels += torch.count_nonzero(out).item()
        return self.cache[index]

    def __len__(self):
        return len(self.img_files)
