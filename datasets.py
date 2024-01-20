# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from email import parser
import os
from random import random
from cv2 import split
import torch
import json
import random

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


from PIL import Image
from modeling_finetune import MAX_CAP_LEN

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, caption_dir, split='train'):
        self.split = split
        self.img_dir = os.path.join(img_dir,self.split+'2017')

        if split == 'train':
            caption_path = os.path.join(caption_dir,'selected_train.json')
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((224,224),scale=(0.2, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
  
            parser_path = os.path.join(caption_dir,'obj2col.json')
            self.parser_file = json.load(open(parser_path,'r'))
        elif split == 'val':
            caption_path = os.path.join(caption_dir,'selected_val.json')    
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
            
        elif split == 'test':
            self.img_dir = 'example'
            caption_path = os.path.join('test','test.json')
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
        self.caption_file = json.load(open(caption_path,'r'))
        self.keys = list(self.caption_file.keys())
                 
    def get_img(self, img_name):
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth).convert('RGB')
        img = self.transform(img)
        return img

    def get_caption(self, key):
        captions = self.caption_file[key]
        index = random.choice([i for i in range(len(captions))])
        cap = captions[index]
        return cap,index
    
    def get_parser(self, key,cap_idx):
        parsers = self.parser_file[key]
        parser = parsers[cap_idx]
        # print(parser)
        occm_gt = torch.zeros((MAX_CAP_LEN,MAX_CAP_LEN)) # [0-19]
        for pair in parser:
            if pair[0]<=MAX_CAP_LEN and pair[1]<=MAX_CAP_LEN:    
                occm_gt[pair[0]-1,pair[1]-1] = 1
        return occm_gt

    def __getitem__(self, index):
        key = self.keys[index]
        img = self.get_img(key)
        cap,cap_idx = self.get_caption(key)
        if self.split == 'test' or self.split == 'val':
            parser_mat = torch.zeros((MAX_CAP_LEN,MAX_CAP_LEN))
        else:
            parser_mat = self.get_parser(key,cap_idx)
        return img, cap, key, parser_mat

    def __len__(self):
        return len(self.keys)