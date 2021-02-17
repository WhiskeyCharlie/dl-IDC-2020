import datetime
import time
from collections import defaultdict
from typing import List

import numpy as np
import torch
import torchvision
from model import UNET
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import glob
import os
import sys
from PIL import Image
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

BATCH_SIZE = 10
IMAGE_SIZE = 128
NUM_EPOCHS = 50
VALID_EVAL = True
RESIZE_ALL = False
TRAIN_MODE = True
CONT_TRAIN = False
OI_DATASET = False
GH_DATASET = True
CONT_MODEL = './saved_models/unet_64x_100e.pt'
EVAL_MODEL = './saved_models/unet_64x_20e_2021-02-17-11-47.pt'
FIRST_CHAR = '0'


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

    def positive_class_weight(self):
        return self.dataset.get_positive_weights()

    def get_positive_weights(self):
        if self.total_pixels == 0:
            return 0
        return self.total_positive_pixels / self.total_pixels

    def __getitem__(self, index):
        return self.images.get(index)

    def __len__(self):
        return len(self.images)


class DatasetSegmentationOI(data.Dataset):
    def __init__(self, main_folder_path, segments_folder_path):
        super(DatasetSegmentationOI, self).__init__()
        self.mask_files = glob.glob(os.path.join(segments_folder_path, '*.png'))
        self.img_files = []
        self.cache = dict()
        self.mask_files_with_corresponding_img = []
        for mask_path in self.mask_files:
            base_file_name = os.path.basename(mask_path)
            img_path = os.path.join(main_folder_path, os.path.basename(mask_path).split('_')[0] + '.jpg')
            # Second clause is so we only take some of the pics
            if os.path.exists(img_path) and base_file_name[0] in FIRST_CHAR:
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
            background = Image.new('1', label.size, 0)
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


def train(model, train_dl, valid_dl, loss_fn, optimizer, scheduler, metrics_fn, epochs=1):
    start = time.time()

    train_loss, valid_loss = [], []
    acc, mcc, inform = defaultdict(list), defaultdict(list), defaultdict(list)
    for _ in tqdm(range(epochs), total=epochs, unit='epoch'):
        scheduler.step()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = torch.tensor([0, 0, 0, 0], dtype=torch.float64)

            # iterate over data
            for x, y in tqdm(dataloader, leave=False):

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)

                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                batch_metrics = metrics_fn(outputs, y)

                # Note: outputs.shape[0] not necessarily batch size, rather the number in this batch (<= batch_size)
                running_acc += batch_metrics * outputs.shape[0]
                running_loss += loss.detach().item() * outputs.shape[0]
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_metrics = running_acc / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}')
            for label, metric in zip(['TPR', 'FPR', 'TNR', 'FNR'], metrics_to_rates(epoch_metrics)):
                print(f'\t>{label}: {metric:0.2f}')
            epoch_metrics_arr = np.asarray(epoch_metrics)
            tp, fp, tn, fn = epoch_metrics_arr
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            print(f'\t>Acc: {accuracy:0.2f}')
            acc[phase].append(accuracy)
            inform[phase].append(informedness(epoch_metrics_arr))
            print(f'\t>Inf: {inform[phase][-1]:0.2f}')
            mcc[phase].append(matthews_correlation_coefficient(epoch_metrics_arr))
            print(f'\t>Mcc: {mcc[phase][-1]:0.2f}')

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
        curr_time = time.time()
        time_elapsed = curr_time - start
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        start = curr_time

    return train_loss, valid_loss, acc, inform, mcc


def informedness(metrics_arr):
    tp, fp, tn, fn = metrics_arr
    return (tp / (tp + fn)) + (tn / (tn + fp)) - 1


def matthews_correlation_coefficient(metrics_arr):
    tp, fp, tn, fn = metrics_arr
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0
    return numerator / denominator


def metrics_to_rates(metrics_tensor: torch.Tensor) -> List[float]:
    arr = np.asarray(metrics_tensor)
    tp, fp, tn, fn = arr
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = 1 - tnr
    fnr = 1 - tpr
    return [tpr, fpr, tnr, fnr]


def metrics(pred_b, y_b) -> torch.Tensor:
    predicted_res = pred_b.clone().detach()
    predicted_res = torch.gt(predicted_res, 0.5).detach()
    bool_pred_b = torch.gt(predicted_res, 0.5)
    int_pred_b = bool_pred_b.int()
    int_y_b = y_b.int()

    true_positive_tensor = int_pred_b & int_y_b
    true_positive = torch.mean(true_positive_tensor.float()).detach().item()

    false_positive_tensor = int_pred_b & (1 - int_y_b)
    false_positive = torch.mean(false_positive_tensor.float()).detach().item()

    true_negative_tensor = (1 - int_pred_b) & (1 - int_y_b)
    true_negative = torch.mean(true_negative_tensor.float()).detach().item()

    false_negative_tensor = (1 - int_pred_b) & int_y_b
    false_negative = torch.mean(false_negative_tensor.float()).detach().mean()

    return torch.tensor([true_positive, false_positive, true_negative, false_negative])


def load_oi_images():
    batch_size = BATCH_SIZE
    train_dataset = DatasetSegmentationOI(f'oid/train_{IMAGE_SIZE}/',
                                          f'oid/train_segments_people_{IMAGE_SIZE}')
    validation_dataset = DatasetSegmentationOI(f'oid/validation_{IMAGE_SIZE}/',
                                               f'oid/validation_segments_people_{IMAGE_SIZE}')
    train_loader = DataLoaderSegmentation(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoaderSegmentation(validation_dataset, batch_size=batch_size, num_workers=0)
    # Ugly hack to initialize the positive class counts.
    for _ in zip(train_loader, test_loader):
        pass
    return train_loader, test_loader


def load_gh_images():
    batch_size = 5
    full_dataset = DatasetSegmentationGH(f'./gh_dataset/train_{IMAGE_SIZE}', f'./gh_dataset/segments_{IMAGE_SIZE}')
    num_train_instances = int(len(full_dataset) * 0.8)
    num_valid_instances = len(full_dataset) - num_train_instances
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset,
                                                                      [num_train_instances, num_valid_instances])

    # Horrible hack, truly
    def pos_instances():
        return full_dataset.get_positive_weights()
    setattr(train_dataset, 'get_positive_weights', pos_instances)
    setattr(validation_dataset, 'get_positive_weights', pos_instances)

    train_loader = DataLoaderSegmentation(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoaderSegmentation(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, test_loader


def evaluate_valid_images(model_path='./unet.pt', evaluate_some_test=False):
    network = UNET(3, 1)
    network.load_state_dict(torch.load(model_path))
    network.eval()
    train_dl, valid_dl = load_oi_images() if OI_DATASET else load_gh_images()
    for valid_img_batch, valid_mask_batch in valid_dl:
        predicted_batch = network(valid_img_batch).gt(0.5)
        for valid_img, valid_mask, predicted_mask in zip(valid_img_batch, valid_mask_batch, predicted_batch):
            grid = ImageGrid(plt.figure(figsize=(4, 4)), 111, nrows_ncols=(2, 2), axes_pad=0.1)
            for ax, im in zip(grid, [valid_img.detach().permute(1, 2, 0),
                                     valid_mask.detach().permute(1, 2, 0),
                                     predicted_mask.detach().permute(1, 2, 0)]):
                ax.imshow(im)
            plt.show()
    ctr = 0
    if evaluate_some_test:
        for valid_img_batch, valid_mask_batch in train_dl:
            if ctr > 100:
                break
            predicted_batch = network(valid_img_batch).gt(0.5)
            for valid_img, valid_mask, predicted_mask in zip(valid_img_batch, valid_mask_batch, predicted_batch):
                grid = ImageGrid(plt.figure(figsize=(4, 4)), 111, nrows_ncols=(2, 2), axes_pad=0.1)
                for ax, im in zip(grid, [valid_img.detach().permute(1, 2, 0),
                                         valid_mask.detach().permute(1, 2, 0),
                                         predicted_mask.detach().permute(1, 2, 0)]):
                    ax.imshow(im)
                plt.show()


def resize_images(src_directory: str, target_directory: str, dimensions=(IMAGE_SIZE, IMAGE_SIZE)):
    src_files = glob.glob(os.path.join(src_directory, '*'))
    for src_img_path in src_files:
        base_src_file_name = os.path.basename(src_img_path)
        target_img_path = os.path.join(target_directory, base_src_file_name)
        src_img_contents = Image.open(src_img_path).convert('RGB')
        src_img_contents = src_img_contents.resize(dimensions)
        src_img_contents.save(target_img_path)


def get_sortable_timestamp():
    return datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')


def plot_train_valid_loss(train_loss: List[float], valid_loss: List[float], plot_path):
    epochs = list(range(1, len(train_loss) + 1))
    plt.plot(epochs, train_loss, 'g', label='Train Loss')
    plt.plot(epochs, valid_loss, 'b', label='Valid Loss')
    plt.title('Training & Validation results')
    plt.xlabel('Epochs')
    plt.xticks([x - 1 for x in epochs[::10]])
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid()
    plt.savefig(plot_path, dpi=300)
    plt.clf()


def plot_metrics(accuracy, inform, mcc, plot_path):
    for phase in ['train', 'valid']:
        accuracy_p, inform_p, mcc_p = accuracy[phase], inform[phase], mcc[phase]
        epochs = list(range(1, len(accuracy_p) + 1))
        plt.plot(epochs, accuracy_p, 'r', label='Accuracy')
        plt.plot(epochs, inform_p, 'g', label='Informedness')
        plt.plot(epochs, mcc_p, 'b', label="Matthew's CC")
        plt.title('Classification Metrics')
        plt.xlabel('Epochs')
        plt.xticks([x - 1 for x in epochs[::10]])
        plt.ylabel('Metrics')
        plt.ylim((-1, 1))
        plt.legend()
        plt.grid()
        plt.savefig(plot_path.replace('.png', f'_{phase}.png'), dpi=300)
        plt.clf()


def main():
    torch.set_num_threads(20)
    print('Available Threads:', torch.get_num_threads())
    if RESIZE_ALL:
        if OI_DATASET:
            resize_images('./oid/train/', f'./oid/train_{IMAGE_SIZE}')
            print('Resized Training')
            resize_images('./oid/validation', f'./oid/validation_{IMAGE_SIZE}')
            print('Resized Validation')
            resize_images('./oid/train_segments_people', f'./oid/train_segments_people_{IMAGE_SIZE}')
            print('Resized Training Segments')
            resize_images('./oid/validation_segments_people', f'./oid/validation_segments_people_{IMAGE_SIZE}')
            print('Resized Validation Segments')
            print('Done')
            exit(0)
        elif GH_DATASET:
            resize_images('./gh_dataset/train/', f'./gh_dataset/train_{IMAGE_SIZE}')
            resize_images('./gh_dataset/segments/', f'./gh_dataset/segments_{IMAGE_SIZE}')
    if TRAIN_MODE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if OI_DATASET:
                train_dl, valid_dl = load_oi_images()
            else:
                train_dl, valid_dl = load_gh_images()
            network = UNET(3, 1)
            if CONT_TRAIN:
                # noinspection PyBroadException
                try:
                    loaded_dict = torch.load(CONT_MODEL)
                    network.load_state_dict(loaded_dict)
                except:
                    print('Training Continuation failed', file=sys.stderr)
                    exit(1)
            pos_weight = train_dl.positive_class_weight()
            pw = torch.FloatTensor([1 / pos_weight])
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9, nesterov=True)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
            train_loss, validation_loss, accuracy, inform, mcc = train(network, train_dl, valid_dl, loss_fn, optimizer,
                                                                       scheduler, metrics, epochs=NUM_EPOCHS)
            cont_str = '_c' if CONT_TRAIN else ''
            model_name = f'./saved_models/unet_{IMAGE_SIZE}x_{NUM_EPOCHS}e_{get_sortable_timestamp()}{cont_str}'
            model_path = f'{model_name}.pt'
            torch.save(network.state_dict(), model_path)
            plot_train_valid_loss(train_loss, validation_loss, f'{model_name}_loss.png')
            plot_metrics(accuracy, inform, mcc, f'{model_name}_metrics.png')
            if VALID_EVAL:
                evaluate_valid_images(model_path)
            exit(0)

    if VALID_EVAL:
        evaluate_valid_images(EVAL_MODEL)


if __name__ == '__main__':
    torch.manual_seed(42)
    main()
