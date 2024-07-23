import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class ImageLoader:
    def __init__(self, path, seed=10, batch_size=64, num_workers=4):
        self.path = path
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = self.get_transforms()
        self.set_seed()

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 320)),  # 224, 224가 아닌 256, 320?
        ])

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_images(self):
        train_dataset = datasets.ImageFolder(root=self.path, transform=self.transform)
        print(f'Found Images: {len(train_dataset)}, images belonging to: {len(train_dataset.classes)}')

        assert len(train_dataset.classes) == 2
        # f"Expected 2 classes, but found {len(train_dataset.classes)}"

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)
        return train_loader

# print(f'Total road images: {len(X)}, Total mask images: {len(Y)}')

# test run
# path = f'C:\\Users\jdah5454\PycharmProjects\Lane Detection Usin FCN\dataset\\tusimple_preprocessed\\training'
# path = f'C:\\Users\jdah5454\PycharmProjects\Lane Detection Usin FCN\dataset\TuSimple_Preprocessed_Dataset\\training\kaggle\working\\tusimple_preprocessed\\training'
# image_loader = ImageLoader(path)
# run = image_loader.load_images()
# run