import os
import pickle
from matplotlib.pyplot import text
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import RobertaTokenizer
from util import text_aug
# import spacy
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
import random
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('/data/jingliqiang/xiecan/MILNet/model/roberta-base')


# 读取image-text
image_text_dir = '/data/jingliqiang/xiecan/MILNet/data/clean_data/clean_image_text.txt'
def get_image_text():
    image_text = {}
    with open(image_text_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            if sp[0] not in image_text.keys():
                image_text[sp[0]] = " ".join(sp[1:])
    return image_text

# 读取text, label
train_dir = '/data/jingliqiang/xiecan/MILNet/data/clean_data/train.txt'
valid_dir = '/data/jingliqiang/xiecan/MILNet/data/clean_data/valid.txt'
test_dir = '/data/jingliqiang/xiecan/MILNet/data/clean_data/test.txt'


aug_text_dir="/data/jingliqiang/xiecan/MILNet/data/clean_data/back_translate_text_tx.txt"
oppoite_dir=""
def get_text():
    all_text = {}
    aug_text={}
    all_label = {}
    with open(valid_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-2]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys():
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label
    with open(test_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-2]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys():
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label
    with open(train_dir) as f:
        for line in f.readlines():
            sp = line.strip().split()
            sp_1 = sp[1:-1]
            sp_1[0] = sp_1[0][1:]
            sp_1[-1] = sp_1[-1][:-2]
            sp_label = int(sp[-1][:1])
            if sp[0][2:-2] not in all_text.keys():
                all_text[sp[0][2:-2]] = " ".join(sp_1)
                all_label[sp[0][2:-2]] = sp_label
    with open(aug_text_dir) as f:
        for line in f.readlines():
            line=line.split('\n')[0]
            id=line.split('[')[1].split(',')[0].split("'")[1]
            seq=line.split('[')[1].split(',')[1]
            if id not in aug_text.keys():
                aug_text[id]=seq
    return all_text, all_label,aug_text

def get_opposite():
    with open("/data/jingliqiang/xiecan/MILNet/data/train_id_2.pkl","rb") as f:
        opposite_id=pickle.load(f)
    with open("/data/jingliqiang/xiecan/MILNet/data/clean_data/opposite_text.pkl","rb") as f:
        opposite_text=pickle.load(f)
    with open("/data/jingliqiang/xiecan/MILNet/data/clean_data/opposite_label.pkl","rb") as f:
        opposite_label=pickle.load(f)
    return opposite_id,opposite_text,opposite_label
def get_ood():
    with open("/data/jingliqiang/xiecan/MILNet/data/clean_data/ood_text_2.pkl","rb") as f:
        ood_text=pickle.load(f)
    with open("/data/jingliqiang/xiecan/MILNet/data/clean_data/ood_label_2.pkl","rb") as f:
        ood_label=pickle.load(f)
    return ood_text,ood_label

class RGDataset(Dataset):
    def __init__(self, id_path,image_path,max_len):
        with open(id_path, "rb") as f:
            self.id = pickle.load(f)
            self.image_path = image_path
            self.all_text_dic, self.all_label_dic,self.aug_text = get_text()
            self.image_text_dic = get_image_text()
            # self.faster_dic = get_faster_rcnn()
            self.max_len = max_len

    # 图片转tensor
    def image_process(self,image_path):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        image = Image.open(image_path)
        image = transform(image)
        return image

    # bert_tokeninzer 处理成input_id，mask
    def text_process(self,text_now):
        #EDA：
        text_now = text_aug(text_now)
        tokens = text_now
        bert_indices = tokenizer(tokens)['input_ids']
        if len(bert_indices) > self.max_len:
            bert_indices = bert_indices[:(self.max_len - 1)]
            bert_indices.append(2)
        len_text = len(bert_indices)
        len_pad = 77 - len_text
        mask = []
        for m in range(len_text):
            mask.append(1)
        for m in range(len_pad):
            mask.append(0)
            bert_indices.append(1)
        bert_mask = torch.tensor(mask)
        bert_indices = torch.tensor(bert_indices)
        return bert_mask, bert_indices

    def __getitem__(self, index):
        name = self.id[index]
        label = self.all_label_dic[name]

        # raw text bert_indices, bert_mask
        text_raw = self.all_text_dic[name]
        bert_mask, bert_indices = self.text_process(text_raw)

        # iamge_text bert_indices, bert_mask
        image_text = ''
        if name in self.image_text_dic:
            image_text_tmp = self.image_text_dic[name]
            image_text = image_text + image_text_tmp

        # added text bert_indices_add, bert_mask_add
        text_add = text_raw + image_text
        
        bert_mask_add, bert_indices_add = self.text_process(text_add)
        

        # image trans
        image_path = os.path.join(self.image_path, str(name) + ".jpg")
        image_trans = self.image_process(image_path)  # 3*224*224p
        
        return bert_mask, bert_mask, bert_indices,image_trans,label
        # return bert_mask, bert_mask_add, bert_indices_add,image_trans,label

    def __len__(self):
        return len(self.id)
    
class RG2Dataset(Dataset):
    def __init__(self, id_path,image_path,max_len):
        with open(id_path, "rb") as f:
            self.id = pickle.load(f)
            self.image_path = image_path
            self.all_text_dic, self.all_label_dic,_ = get_text()
            self.image_text_dic = get_image_text()
            self.oppoite_id,self.oppoite_text_dic,self.opposite_label_dic=get_opposite()
            self.ood_text_dic,self.ood_label_dic=get_ood()
            self.max_len = max_len

    # 图片转tensor
    def image_process(self,image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        image = Image.open(image_path)
        image = transform(image)
        return image

    # bert_tokeninzer 处理成input_id，mask
    def text_process(self,text_now):
        tokens = text_now
        bert_indices = tokenizer(tokens)['input_ids']
        if len(bert_indices) > self.max_len:
            bert_indices = bert_indices[:(self.max_len - 1)]
            bert_indices.append(2)
        len_text = len(bert_indices)
        len_pad = 77 - len_text
        mask = []
        for m in range(len_text):
            mask.append(1)
        for m in range(len_pad):
            mask.append(0)
            bert_indices.append(1)
        bert_mask = torch.tensor(mask)
        bert_indices = torch.tensor(bert_indices)
        return bert_mask, bert_indices


    def __getitem__(self, index):
        name = self.id[index]
        #opposite
        if 'b' in name:
            name=name.split('b')[1]
            label = self.opposite_label_dic[name]

            # raw text bert_indices, bert_mask
            text_raw = self.oppoite_text_dic[name]
            
        #ood数据集
        elif 'o' in name:
            label = self.ood_label_dic[name]

            # raw text bert_indices, bert_mask
            text_raw = self.ood_text_dic[name]
            name=name.split('o')[1]
        else:
            label = self.all_label_dic[name]

            # raw text bert_indices, bert_mask
            text_raw = self.all_text_dic[name]
            
        bert_mask, bert_indices = self.text_process(text_raw)

        # iamge_text bert_indices, bert_mask
        image_text = ''
        if name in self.image_text_dic:
            image_text_tmp = self.image_text_dic[name]
            image_text = image_text + image_text_tmp

        # added text bert_indices_add, bert_mask_add
        text_add = text_raw + image_text
        bert_mask_add, bert_indices_add = self.text_process(text_add)

        # image trans
        image_path = os.path.join(self.image_path, str(name) + ".jpg")
        image_trans = self.image_process(image_path)  # 3*224*224p


        # return bert_mask, bert_mask_add, bert_indices_add,image_trans,label
        return bert_mask, bert_mask, bert_indices,image_trans,label

    def __len__(self):
        return len(self.id)  
    
    
      
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset=AugDataset(os.path.join("/data/jingliqiang/xiecan/MILNet2/data/",'train_id.pkl'),"/data/jingliqiang/xiecan/data/sarcasm-detection/dataset_image",77)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                    num_workers=8, pin_memory=True)

