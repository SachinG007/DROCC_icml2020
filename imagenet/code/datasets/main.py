from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .imagenet10 import ImageNet10_Dataset

def load_dataset(dataset_name, data_path_train, data_path_test, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'imagenet')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path_train, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path_train, normal_class=normal_class)
    
    if dataset_name == 'imagenet':
        dataset = ImageNet10_Dataset(root_train=data_path_train, root_test = data_path_test, normal_class=normal_class)
    return dataset
