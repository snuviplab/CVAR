import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class Movielens1MbaseDataset(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, content, description, device):
        super(Movielens1MbaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {}
        for name in df.columns:
            self.name2array[name] = torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device)
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.features.extend(["video", "text"])
        self.label = 'rating'
        self.device = device
        self.content = content

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'pretrained':
                pass
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))
                
    def __getitem__(self, index):
        x_dict = {}
        item_id = self.name2array["item_id"][index].item()
        video = self.content[item_id]["video"]
        x_dict["video"] = torch.from_numpy(video).to(self.device).to(torch.float32)
        text = self.content[item_id]["text"]
        x_dict["text"] = torch.from_numpy(text).to(self.device).to(torch.float32)
        for name in self.features:
            if name == "video" or name == "text":
                continue
            else:
                x_dict[name] = self.name2array[name][index]
        return x_dict, self.name2array[self.label][index].squeeze()

    def __len__(self):
        return self.length


class MovieLens1MColdStartDataLoader(object):
    """
    Load all splitted MovieLens 1M Dataset for cold start setting

    :param dataset_path: MovieLens dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, shuffle=True):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        self.content = data["content_features"]
        for key, df in data.items():
            if key in ["warm_test", "description", "content_features"]:
                continue
            if 'metaE' not in key:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.content, self.description, device), batch_size=bsz, shuffle=shuffle)
            else:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=False)
        self.keys = list(self.dataloaders.keys())
        self.item_features = [desc[0] for desc in self.description if desc[0] not in ["user_id", "rating"]]
                                

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]
