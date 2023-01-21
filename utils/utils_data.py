import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import cv2
import numpy as np

from utils import utils_image as ui


def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in
               [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


# 데이터 셋 클래스
class Dataset:
    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class NormalDataset(Dataset):
    def __init__(self, images_dir, image_size, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(image_size // upscale_factor, image_size // upscale_factor),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()])

        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()])

    # fot iterator
    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)


class RandomDegradationDataset(Dataset):
    def __init__(self, images_dir, image_size, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.image_size = image_size
        self.upscale_factor = upscale_factor

    def __getitem__(self, idx):
        image = ui.uint2single(ui.imread_uint(self.filenames[idx], 3))

        lr_image, hr_image = ui.degradation_bsrgan(
            image,
            scale_factor=self.upscale_factor,
            lr_size=self.image_size//self.upscale_factor)

        lr = self.to_tensor(lr_image)
        hr = self.to_tensor(hr_image)

        return lr, hr

    def __len__(self):
        return len(self.filenames)


class TestDataset(Dataset):
    def __init__(self, images_dir, upscale_factor):
        self.filenames = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if check_image_file(x)]
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.upscale_factor = upscale_factor

    # fot iterator
    def __getitem__(self, idx):
        hr_image = Image.open(self.filenames[idx]).convert("RGB")
        hr_image = np.array(hr_image)

        size = hr_image.shape
        hr_image = cv2.resize(hr_image, dsize=(size[1] - size[1] % self.upscale_factor, size[0] - size[0] % self.upscale_factor))
        lr_image = cv2.resize(hr_image, dsize=(0, 0), fx=1/self.upscale_factor, fy=1/self.upscale_factor, interpolation=cv2.INTER_CUBIC)

        hr = self.to_tensor(hr_image)
        lr = self.to_tensor(lr_image)
        return lr, hr

    def __len__(self):
        return len(self.filenames)


def set_dataloader(data_dir, image_size=256, upscale_factor=4, aug_factor=1, batch_size=1,
                   datatype='train', random_degradation=False):

    datatype_dir = os.path.join(data_dir, f'{datatype}_HR')

    image_size = calculate_valid_crop_size(image_size, upscale_factor)
    dataset = get_dataset(images_dir=datatype_dir, image_size=image_size, upscale_factor=upscale_factor,
                          datatype=datatype, random_degradation=random_degradation)
    if datatype == 'train':
        dataset = ConcatDataset([dataset] * aug_factor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size
    )

    return dataloader


def get_dataset(images_dir, image_size, upscale_factor, datatype, random_degradation):
    if datatype == 'test':
        return TestDataset(images_dir, upscale_factor)
    elif random_degradation:
        return RandomDegradationDataset(images_dir, image_size, upscale_factor)
    else:
        return NormalDataset(images_dir, image_size, upscale_factor)
