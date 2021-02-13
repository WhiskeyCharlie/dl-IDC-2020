import time

import torch
import torchvision
from model import UNET
import torch.utils.data as data
import glob
import os
from PIL import Image


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, main_folder_path, segments_folder_path):
        super(DataLoaderSegmentation, self).__init__()
        self.mask_files = glob.glob(os.path.join(segments_folder_path, '*.png'))
        self.img_files = []
        self.mask_files_with_corresponding_img = []
        for mask_path in self.mask_files:
            img_path = os.path.join(main_folder_path, os.path.basename(mask_path).split('_')[0] + '.jpg')
            if os.path.exists(img_path):
                self.img_files.append(img_path)
                self.mask_files_with_corresponding_img.append(mask_path)
        self.mask_files = None

    def __getitem__(self, index):
        IMG_SIZE = 64
        img_path = self.img_files[index]
        mask_path = self.mask_files_with_corresponding_img[index]
        x_img = Image.open(img_path).convert("RGB")
        x_img = x_img.resize((IMG_SIZE, IMG_SIZE))
        label = Image.open(mask_path).convert("RGB")
        # TODO: this will certainly introduce bugs
        background = Image.new('RGB', label.size, (255, 255, 255))
        background.paste(label)
        background = background.resize((IMG_SIZE, IMG_SIZE))
        image_1 = torchvision.transforms.PILToTensor()(x_img)
        image_2 = torchvision.transforms.PILToTensor()(background)
        return image_1, image_2

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

            step = 0

            # iterate over data
            for x, y in dataloader:
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x.float())
                    loss = loss_fn(outputs.squeeze(1), y.float().squeeze(1))

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def acc_metric(pred_b, yb):
    # return (pred_b.argmax(dim=1) == yb).float().mean()
    return 0


def load_images():
    train_dataset = DataLoaderSegmentation('oid/train/', 'oid/train_segments_people')
    validation_dataset = DataLoaderSegmentation('oid/validation', 'oid/validation_segments_people')
    return data.DataLoader(train_dataset, batch_size=1), data.DataLoader(validation_dataset, batch_size=1)


def main():
    train_dl, valid_dl = load_images()
    network = UNET(3, 3)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    train_loss, validation_loss = train(network, train_dl, valid_dl, loss_fn, optimizer, acc_metric,
                                        epochs=50)
    print(train_loss, validation_loss)


if __name__ == '__main__':
    main()
