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


class SignalDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.item_path = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        item_name = self.item_path[idx]
        item_path = os.path.join(self.root_dir, self.label_dir, item_name)
        signal_data = read_matrix_file(item_path)
        # transform
        signal_data = Image.fromarray(signal_data)
        signal_data = self.transform(signal_data)
        label = int(self.label_dir.split('_')[1]) - 1
        return signal_data, label

    def __len__(self):
        return len(self.item_path)


def dataset_joint(root_dir, transform):
    filenames = os.listdir(os.path.join(root_dir))
    dataset = None
    for filename in filenames:
        sig_label = filename
        sig_dataset = SignalDataset(root_dir, sig_label, transform)
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

    data = read_matrix_file("/home/wanghao/频域数据/signal_python_500B_6/signal_1/1_sign0_f029.9298_A1.3632.txt")

    train_dir = '/home/wanghao/频域数据/signal_python_500B_6_split/train'
    test_dir = '/home/wanghao/频域数据/signal_python_500B_6_split/test'

    train_dataset = dataset_joint(train_dir, train_transforms)
    test_dataset = dataset_joint(test_dir, test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    for idx, data in enumerate(test_loader):
        print(idx, data[0].shape, data[1].shape)

