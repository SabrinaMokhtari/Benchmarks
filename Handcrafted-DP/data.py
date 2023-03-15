import torch
from torchvision import datasets, transforms
from kymatio.torch import Scattering2D
import os
import pickle
import numpy as np
import logging
import torchvision
from torch.utils.data import DataLoader, Dataset

from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
from libauc.models import densenet121 as DenseNet121
from libauc.datasets import CheXpert
from PIL import Image
from sklearn.metrics import roc_auc_score

import cv2
import torchvision.transforms as tfs
import pandas as pd

SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1),
    "chest_xray": (64, 64, 3),
    "eye": (64, 64, 3),
    "chexpert": (64, 64, 3)
}


class MyImageDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, augment=None):
        self.root = root
        self.augment = augment
        self.transform = transform
        self.target_transform = target_transform
        self.image_folder = torchvision.datasets.ImageFolder(self.root)

    def __len__(self):
        return len(self.image_folder.imgs)

    def __getitem__(self, idx):
        path, target = self.image_folder.imgs[idx]
        img = self.image_folder.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __getshape__(self):
        print("HI:\n")
        return (self.__len__(), *self.__getitem__(0)[0].shape)


class CheXpert(Dataset):
    '''
    Reference:
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''

    def __init__(self,
                 csv_path,
                 image_root_path='',
                 image_size=320,
                 class_index=0,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 mode='train'):

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

            # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

            # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]

        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:  # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size

        self._images_list = [image_root_path + path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                print('-' * 30)
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                    self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    print('Found %s images in total, %s positive images, %s negative images' % (
                    self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                print('-' * 30)
            else:
                print('-' * 30)
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    imratio = self.value_counts_dict[class_key][1] / (
                                self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    imratio_list.append(imratio)
                    print('Found %s images in total, %s positive images, %s negative images' % (
                    self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                    print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                    print()
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                print('-' * 30)

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def image_augmentation(self, image):
        img_aug = transforms.Compose([transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                                                              scale=(0.95, 1.05),
                                                              fill=128)])  # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train':
            image = self.image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ = np.array([[[0.229, 0.224, 0.225]]])
        image = (image - __mean__) / __std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1:  # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label


def get_scatter_transform(dataset):
    shape = SHAPES[dataset]
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h // 4, w // 4)


def get_data(name, augment=False, **kwargs):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR10(root=".data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root=".data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))

    elif name == "fmnist":
        train_set = datasets.FashionMNIST(root='.data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='.data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

    elif name == "mnist":
        train_set = datasets.MNIST(root='.data', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='.data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    elif name == "cifar10_500K":

        # extended version of CIFAR-10 with pseudo-labelled tinyimages

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = SemiSupervisedDataset(kwargs['aux_data_filename'],
                                          root=".data",
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(train_transforms))
        test_set = None

    elif name == "chest_xray":
        main_dir = '/content/drive/MyDrive/CS860/Project/Kaggle/chest_xray'
        train_dir = main_dir + '/train'
        test_dir = main_dir + '/test'

        trans = transforms.Compose(transforms=[transforms.Resize((64, 64)), transforms.ToTensor()])
        train_set = MyImageDataset(root=train_dir, transform=trans)
        test_set = MyImageDataset(root=test_dir, transform=trans)

    elif name == "eye":
        main_dir = '/content/drive/MyDrive/CS860/Project/Kaggle/OCT2017 '
        train_dir = main_dir + '/train'
        test_dir = main_dir + '/test'

        trans = transforms.Compose(transforms=[transforms.Resize((64, 64)), transforms.ToTensor()])
        train_set = MyImageDataset(root=train_dir, transform=trans)
        test_set = MyImageDataset(root=test_dir, transform=trans)

    elif name == "chexpert":
        root = '/home/ubuntu/chexpertchestxrays-u20210408/CheXpert-v1.0/'

        # Index: -1 denotes multi-label mode including 5 diseases
        train_set = CheXpert(csv_path=root + 'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True,
                             image_size=64, mode='train', class_index=-1)
        test_set = CheXpert(csv_path=root + 'valid.csv', image_root_path=root, use_upsampling=False, use_frontal=True,
                            image_size=64, mode='valid', class_index=-1)

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 aux_data_filename=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        self.dataset = datasets.CIFAR10(train=train, **kwargs)
        self.train = train

        # shuffle cifar-10
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.targets = list(np.asarray(self.targets)[p])

        if self.train:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            aux_path = os.path.join(kwargs['root'], aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']
            orig_len = len(self.data)

            # shuffle additional data
            p = np.random.permutation(len(aux_data))
            aux_data = aux_data[p]
            aux_targets = aux_targets[p]

            self.data = np.concatenate((self.data, aux_data), axis=0)
            self.targets.extend(aux_targets)

            # note that we use unsup indices to track the labeled datapoints
            # whose labels are "fake"
            self.unsup_indices.extend(
                range(orig_len, orig_len + len(aux_data)))

            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, num_batches, batch_size):
        self.inds = list(range(num_examples))
        self.batch_size = batch_size
        self.num_batches = num_batches
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        inds_shuffled = [self.inds[i] for i in torch.randperm(len(self.inds))]

        while len(inds_shuffled) < self.num_batches * self.batch_size:
            temp = [self.inds[i] for i in torch.randperm(len(self.inds))]
            inds_shuffled.extend(temp)

        for k in range(0, self.num_batches * self.batch_size, self.batch_size):
            if batch_counter == self.num_batches:
                break

            batch = inds_shuffled[k:(k + self.batch_size)]

            # this shuffle operation is very important, without it
            # batch-norm / DataParallel hell ensues
            np.random.shuffle(batch)
            yield batch
            batch_counter += 1

    def __len__(self):
        return self.num_batches


class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def get_scattered_dataset(loader, scattering, device, data_size):
    # pre-compute a scattering transform (if there is one) and return
    # a TensorDataset

    scatters = []
    targets = []

    num = 0
    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

        num += len(data)
        if num > data_size:
            break

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    scatters = scatters[:data_size]
    targets = targets[:data_size]

    data = torch.utils.data.TensorDataset(scatters, targets)
    return data


def get_scattered_loader(loader, scattering, device, drop_last=False, sample_batches=False):
    # pre-compute a scattering transform (if there is one) and return
    # a DataLoader

    scatters = []
    targets = []

    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        if scattering is not None:
            data = scattering(data)
        scatters.append(data)
        targets.append(target)

    scatters = torch.cat(scatters, axis=0)
    targets = torch.cat(targets, axis=0)

    data = torch.utils.data.TensorDataset(scatters, targets)

    if sample_batches:
        sampler = PoissonSampler(len(scatters), loader.batch_size)
        return torch.utils.data.DataLoader(data, batch_sampler=sampler,
                                           num_workers=0, pin_memory=False)
    else:
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
        return torch.utils.data.DataLoader(data,
                                           batch_size=loader.batch_size,
                                           shuffle=shuffle,
                                           num_workers=0,
                                           pin_memory=False,
                                           drop_last=drop_last)