import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class YahooDataset(Dataset):
    """
    Load a base Yahoo Dataset 
    """
    def __init__(self, dataset_name, df, content, content_mode, description, device):
        super(YahooDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {}
        for name in df.columns:
            self.name2array[name] = torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device)
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.content_mode = content_mode
        if content_mode == "video_only":
            self.features.append("video")
        elif content_mode == "text_only":
            self.features.append("text")
        else:
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
        if self.content_mode == "video_only":
            video = self.content[item_id]["video"]
            x_dict["video"] = torch.from_numpy(video).to(self.device).to(torch.float32)
        elif self.content_mode == "text_only":
            text = self.content[item_id]["text"]
            x_dict["text"] = torch.from_numpy(text).to(self.device).to(torch.float32)
        else:
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


class ColdDataset(Dataset):
    def __init__(self, dataset_name, df, content, content_mode,description, device):
        super(ColdDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.content = content
        self.content_mode = content_mode
        self.description = description
        self.device = device
        self.item_features = self.get_item_features()
        self.num_items = len(self.item_features)
        self.user2items = df.groupby("user_id")["item_id"].apply(list).reset_index(name="item_pos")

    def get_item_features(self):
        item_features = self.df.groupby("item_id", as_index=False).first()[["item_id", "count"]]
        text_features = []
        video_features = []
        for item in item_features["item_id"]:
            text_features.append(self.content[item]["text"])
            video_features.append(self.content[item]["video"])
        if self.content_mode == "video_only":
            item_features["video"] = video_features  
        elif self.content_mode == "text_only":
            item_features["text"] = text_features
        else:
            item_features["video"] = video_features  
            item_features["text"] = text_features
        return item_features
    
    def __getitem__(self, index):
        user_id = self.user2items["user_id"].iloc[index]
        item_pos = self.user2items["item_pos"].iloc[index]
        x_dict = {}
        for name in self.item_features.columns:
            x_dict[name] = torch.from_numpy(np.array(list(self.item_features[name])).reshape([self.num_items, -1])).to(self.device)
        x_dict["user_id"] = torch.from_numpy(np.repeat(user_id, self.num_items).reshape([self.num_items, -1])).to(self.device)
        self.format(x_dict)
        return x_dict, item_pos

    def format(self, x_dict):
        for name, size, type in self.description:
            if type == 'spr' or type == 'seq':
                x_dict[name] = x_dict[name].to(torch.long)
            elif type == 'ctn' or type == 'pretrained':
                x_dict[name] = x_dict[name].to(torch.float32)
            elif type == 'label':
                pass
            else:
                raise ValueError('unknwon type {}'.format(type))
        return x_dict

    def __len__(self):
        return len(self.user2items)


class YahooColdStartDataLoader(object):
    """
    Load all splitted Yahoo Dataset for cold start setting

    :param dataset_path: Yahoo dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, content_mode="all", shuffle=True):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        exclude_col = []
        if content_mode == "video_only":
            exclude_col.append("text")
        elif content_mode == "text_only":
            exclude_col.append("video")
        self.description = [desc for desc in data["description"] if desc[0] not in exclude_col]
        self.content = data["content_features"]
        for key, df in data.items():
            if key.startswith("cold"):
                self.dataloaders[key] = DataLoader(ColdDataset(dataset_name, df, self.content, content_mode, self.description, device), batch_size=1, shuffle=False)
            elif key in ["warm_test", "description", "content_features"]:
                continue
            elif 'metaE' not in key:
                self.dataloaders[key] = DataLoader(YahooDataset(dataset_name, df, self.content, content_mode, self.description, device), batch_size=bsz, shuffle=shuffle)
            else:
                self.dataloaders[key] = DataLoader(YahooDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=False)
        self.keys = list(self.dataloaders.keys())
        self.item_features = [desc[0] for desc in self.description if desc[0] not in ["user_id", "rating"]]
                                

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]
