import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

class Tripletdataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    """

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.cls_to_imgs = {}
        self.imgs_to_cls = {}
        self.all_classes = []
        self.all_train_data = []

        for cls_name in sorted(os.listdir(root_dir)):
            img_list = sorted(os.listdir(os.path.join(root_dir, cls_name)))
            img_path_list = [os.path.join(root_dir,cls_name, img_name) for img_name in img_list]
            self.cls_to_imgs[cls_name] = img_path_list
            self.imgs_to_cls.update({img: cls_name for img in img_path_list})
            self.all_train_data.extend(img_path_list)
        
        self.all_classes = sorted(self.cls_to_imgs.keys())


    def __getitem__(self, index):
        anc_img_path = self.all_train_data[index]
        pos_img_path = np.random.choice(self.cls_to_imgs[self.imgs_to_cls[anc_img_path]])
        neg_candidates = [img for img in self.all_train_data if img not in self.cls_to_imgs[self.imgs_to_cls[anc_img_path]]]
        neg_img_path = np.random.choice(neg_candidates)

        
        anc_img = self.deal_img(Image.open(anc_img_path))
        pos_img = self.deal_img(Image.open(pos_img_path))
        neg_img = self.deal_img(Image.open(neg_img_path))

        if self.transform is not None:
            anc_img = self.transform(anc_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
            # print(anc_img.max(), anc_img.min())
            # print(pos_img.max(), pos_img.min())
            # print(neg_img.max(), neg_img.min())
        # return anc_img, pos_img, neg_img, anc_img_path, pos_img_path, neg_img_path
        label_str = self.imgs_to_cls[anc_img_path]
        label_idx = self.all_classes.index(label_str)
        label = torch.tensor(label_idx, dtype=torch.int64)
        return anc_img, pos_img, neg_img, label

    def __len__(self):
        return len(self.all_train_data)

    # 处理jpg/png图片透明部分
    def deal_img(self, img):
        # 处理png图片透明部分
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 白色背景
            img = Image.alpha_composite(background, img)
            img = img.convert("RGB")  # 转为RGB
        else:
            img = img.convert("RGB")
        return img

# --------------Test code----------------
import matplotlib.pyplot as plt
def show_triplet_samples(dataloader, num_samples=5):
    count = 0
    for batch in dataloader:
        # anc_imgs, pos_imgs, neg_imgs, _, _, neg_img_paths = batch
        anc_imgs, pos_imgs, neg_imgs = batch
        batch_size = anc_imgs.size(0)
        for i in range(batch_size):
            if count >= num_samples:
                return
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(anc_imgs[i].squeeze(), cmap='gray')
            axs[0].set_title('Anchor')
            axs[1].imshow(pos_imgs[i].squeeze(), cmap='gray')
            axs[1].set_title('Positive')
            axs[2].imshow(neg_imgs[i].squeeze(), cmap='gray')
            # axs[2].set_title('Negative, ' + os.path.basename(neg_img_paths[i]))
            axs[2].set_title('Negative')
            for ax in axs:
                ax.axis('off')
            plt.show()
            count += 1

if __name__ == "__main__":
    dataset = Tripletdataset(root_dir="./BI", transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Dataset size: {dataloader.__len__}")
    show_triplet_samples(dataloader, num_samples=5)
