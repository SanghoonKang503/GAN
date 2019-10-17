import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sang_main import *

def get_train_loader(data_dir, batch_sizes, image_size):
    # put image data into data_loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data_dir = 'resized_celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_sizes, shuffle=True)

    # confrimed input image size!
    temp = plt.imread(train_loader.dataset.imgs[0][0])
    if (temp.shape[0] != opt.img_size) or (temp.shape[0] != opt.img_size):
        raise ValueError('image size is not 64 x 64!')
    
    return train_loader

