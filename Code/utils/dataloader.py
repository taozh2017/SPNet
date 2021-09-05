from torch.utils import data
import torch
import os
from PIL import Image
class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        
        self.image_path = list(map(lambda x: os.path.join(img_root, x), sorted(os.listdir(img_root))))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), sorted(os.listdir(label_root))))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')

        gt = Image.open(self.label_path[item]).convert('L')
        # print(self.image_path[item], self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
    