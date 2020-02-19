from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
from random import sample 
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ImageNet10_Dataset(TorchvisionDataset):

    def __init__(self, root_train: str, root_test: str, normal_class=5):
        super().__init__(root_train)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        train_dataset = datasets.ImageFolder(root_train,
                                        transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]), target_transform=target_transform)

        # train_set = MyCIFAR10(root=self.root, train=True, download=True,
        #                       transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        #import pdb;pdb.set_trace()
        train_idx_normal = get_target_label_idx(train_dataset.targets, self.normal_classes)
        # train_idx_normal_train = sample(train_idx_normal, 4000)
        # val_idx_normal = [x for x in train_idx_normal if x not in train_idx_normal_train]

        # rest_train_classes = get_target_label_idx(train_set.train_labels, self.outlier_classes)
        # rest_train_classes_subset = sample(rest_train_classes, 9000)
        # val_idx = val_idx_normal + rest_train_classes_subset
        self.train_set = Subset(train_dataset, train_idx_normal)

        # self.test_set = Subset(train_set, val_idx)
        #self.test_set = MyCIFAR10(root=self.root, train=False, download=True,
        #                          transform=transform, target_transform=target_transform)
        self.test_set = datasets.ImageFolder(root_test,
                                        transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]), target_transform=target_transform)

 
class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
