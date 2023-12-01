from torch.utils.data import Dataset
import os
import numpy as np
import math
import torch
from torchvision import transforms
from PIL import Image


def read_matrix_file(file_path):
    temp = np.loadtxt(file_path, delimiter=',')
    return temp

class SignalDataset_multi_stage(Dataset):
    def __init__(self, root_dir, root_dir2, label_dir, transform):
        self.root_dir = root_dir
        self.root_dir2 = root_dir2
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.item_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        item_name = self.item_path[idx]
        item_path = os.path.join(self.root_dir, self.label_dir, item_name)
        item_path2 = os.path.join(self.root_dir2, self.label_dir, item_name)
        signal_data1 = read_matrix_file(item_path)
        signal_data2 = read_matrix_file(item_path2)
        # transform
        signal_data1 = Image.fromarray(signal_data1)
        signal_data1 = self.transform(signal_data1)
        signal_data2 = Image.fromarray(signal_data2)
        signal_data2 = self.transform(signal_data2)
        label = int(self.label_dir.split('_')[1]) - 1
        return tuple([signal_data1, signal_data2]), label

    def __len__(self):
        return len(self.item_path)


def dataset_joint(root_dir, root_dir2, transform):
    filenames = os.listdir(os.path.join(root_dir))
    dataset = None
    for filename in filenames:
        sig_label = filename
        sig_dataset = SignalDataset_multi_stage(root_dir, root_dir2, sig_label, transform)
        if filename==filenames[0]:
            dataset = sig_dataset
        else:
            dataset = dataset + sig_dataset
    return dataset


train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])

test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224])
        ])


if __name__ == '__main__':

    train_dir = '/home/wanghao/频域数据/signal_python_2000B_1of4_split/train'
    test_dir = '/home/wanghao/频域数据/signal_python_2000B_1of4_split/test'
    train_dir2 = '/home/wanghao/频域数据/signal_python_2000B_3of4'
    test_dir2 = '/home/wanghao/频域数据/signal_python_2000B_3of4'

    train_dataset = dataset_joint(train_dir, train_dir2, train_transforms)
    test_dataset = dataset_joint(test_dir, test_dir2, test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    for idx, data in enumerate(test_loader):
        print(idx, data[0][0].shape, data[0][1].shape, data[1].shape)

