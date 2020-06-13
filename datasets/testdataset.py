import torch as t
from torch.utils.data import Dataset
import random
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os

class testdataset(Dataset):
    def __init__(self, trans = None):
        self.path = "../Desktop/car_images/test"
        self.label = []
        self.imgs = []
        self.class_labels = []
        self.trans = trans
        f = [i for i in os.listdir(self.path)]
        for i in f:
            cat = i[-6]
            self.label.append((i, cat))
            self.class_labels.append(int(cat))
            self.imgs.append(os.path.join(self.path, i))
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        res = Image.open(self.imgs[index])
        res = self.trans(res)
        if random.random() < 0.4:
                too = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(degrees=7),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                res = too(res)
        return res, self.class_labels[index]