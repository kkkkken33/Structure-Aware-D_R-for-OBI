import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image

def get_transform():
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    return transform

class OBI2BIDataset(Dataset):
    def __init__(self, root_dir, json_file, transform):
        super(OBI2BIDataset, self).__init__()
        self.training_data = []
        self.transform = transform
        
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                obc_img = os.path.join(root_dir, data['obc_img'])
                bi_img = os.path.join(root_dir, data['bi_img'])
                obc_text_description = data['obc_text_description']
                self.training_data.append((obc_img, bi_img, obc_text_description))

        
    def __getitem__(self, index):
        obc_img_path, bi_img_path, obc_text_description = self.training_data[index]
        obc_img = self.deal_img(Image.open(obc_img_path))
        bi_img = self.deal_img(Image.open(bi_img_path))

        if self.transform:
            obc_img = self.transform(obc_img)
            bi_img = self.transform(bi_img)
        
        return obc_img, bi_img, obc_text_description
    
    def deal_img(self, img):
        # 处理png图片透明部分
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 白色背景
            img = Image.alpha_composite(background, img)
            img = img.convert("RGB")  # 转为RGB
        else:
            img = img.convert("RGB")
        return img

    def __len__(self):
        return len(self.training_data)
    
if __name__ == "__main__":
    # 假设你的json文件和图片路径都在当前目录下
    root_dir = "./dataset"  
    json_file = "./dataset/dataset_copy.json"
    transform = get_transform()
    dataset = OBI2BIDataset(root_dir, json_file, transform)

    print(f"数据集样本数: {len(dataset)}")

    # 随机取前3个样本，检查图片shape和描述
    for i in range(min(3, len(dataset))):
        obc_img, bi_img, obc_text_description = dataset[i]
        print(f"样本{i}:")
        print(f"  OBC图片shape: {obc_img.shape}")
        print(f"  BI图片shape: {bi_img.shape}")
        print(f"  文本描述: {obc_text_description}...")  # 只显示前50字