import time

import torch
import torchvision
from model import UNET
import torch.utils.data as data
import glob
import os
from PIL import Image
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

BATCH_SIZE = 32
IMAGE_SIZE = 64
NUM_EPOCHS = 100
VALID_EVAL = True
RESIZE_ALL = False


class DataLoaderSegmentation(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderSegmentation, self).__init__(*args, **kwargs)

    def positive_class_weight(self):
        return self.dataset.get_positive_weights()


class DatasetSegmentation(data.Dataset):
    def __init__(self, main_folder_path, segments_folder_path):
        super(DatasetSegmentation, self).__init__()
        self.mask_files = glob.glob(os.path.join(segments_folder_path, '*.png'))
        self.img_files = []
        self.cache = dict()
        self.mask_files_with_corresponding_img = []
        for mask_path in self.mask_files:
            base_file_name = os.path.basename(mask_path)
            img_path = os.path.join(main_folder_path, os.path.basename(mask_path).split('_')[0] + '.jpg')
            # Second clause is so we only take some of the pics
            if os.path.exists(img_path) and base_file_name[0] in '0123':
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
            # if index < 10:
            #     print('miss', len(self.cache))
            img_path = self.img_files[index]
            mask_path = self.mask_files_with_corresponding_img[index]
            x_img = Image.open(img_path).convert("RGB")
            label = Image.open(mask_path).convert("1")
            # TODO: this will certainly introduce bugs
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


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()

    train_loss, valid_loss = [], []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            # iterate over data
            for idx, (x, y) in enumerate(dataloader):

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y.float())

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.float())

                acc = acc_fn(outputs, y)

                # Note: outputs.shape[0] not necessarily batch size, rather the number in this batch (<= batch_size)
                running_acc += acc * outputs.shape[0]
                running_loss += loss.detach().item() * outputs.shape[0]
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc}')

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
        curr_time = time.time()
        time_elapsed = curr_time - start
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        start = curr_time

    return train_loss, valid_loss


def acc_metric(pred_b, yb):
    predicted_res = pred_b.clone().detach()
    predicted_res -= pred_b.min(1, keepdim=True)[0]
    predicted_res /= predicted_res.max(1, keepdim=True)[0]
    predicted_res = torch.gt(predicted_res, 0.5).detach()
    bool_pred_b = torch.gt(predicted_res, 0.5)
    comparison = (bool_pred_b == yb).float()
    acc = torch.mean(comparison)

    return acc.detach().item()


def load_images():
    batch_size = BATCH_SIZE
    train_dataset = DatasetSegmentation('oid/train_64/', 'oid/train_segments_people_64')
    validation_dataset = DatasetSegmentation('oid/validation_64/', 'oid/validation_segments_people_64')
    train_loader = DataLoaderSegmentation(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = DataLoaderSegmentation(validation_dataset, batch_size=batch_size, num_workers=0)
    # Ugly hack to initialize the positive class counts.
    for _ in zip(train_loader, test_loader):
        pass
    return train_loader, test_loader


def evaluate_valid_images():
    network = UNET(3, 1)
    network.load_state_dict(torch.load('./unet.pt'))
    network.eval()
    train_dl, valid_dl = load_images()
    for valid_img_batch, valid_mask_batch in valid_dl:
        predicted_batch = network(valid_img_batch).gt(0.01)
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


def main():
    if RESIZE_ALL:
        resize_images('./oid/train/', f'./oid/train_{IMAGE_SIZE}')
        resize_images('./oid/validation', f'./oid/validation_{IMAGE_SIZE}')
        resize_images('./oid/train_segments_people', f'./oid/train_segments_people_{IMAGE_SIZE}')
        resize_images('./oid/validation_segments_people', f'./oid/validation_segments_people_{IMAGE_SIZE}')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_dl, valid_dl = load_images()
        network = UNET(3, 1)
        pos_weight = train_dl.positive_class_weight()
        pw = torch.FloatTensor([1 / pos_weight])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        train_loss, validation_loss = train(network, train_dl, valid_dl, loss_fn, optimizer, acc_metric,
                                            epochs=NUM_EPOCHS)
        print(train_loss, validation_loss)
        torch.save(network.state_dict(), './unet.pt')

    if VALID_EVAL:
        evaluate_valid_images()


if __name__ == '__main__':
    torch.manual_seed(42)
    main()
