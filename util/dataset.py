from PIL import Image
import os
import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils

def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset == 'cifar10':
        data_path = './data/cifar10'
        args.num_classes = 10
        args.model = 'resnet18'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = './data/cifar100'
        args.num_classes = 100
        # args.model = 'resnet34'
        args.model = 'resnet34'
        
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = iid_sampling(n_train, args.num_users, args.seed)
        return dataset_train, dataset_test, dict_users
        
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)
        return dataset_train, dataset_test, dict_users
