import torch
import numpy as np
from PIL import Image
import os
from OBI_BI_dataset import Tripletdataset
from OBI_BI_dataset import get_transform
from torch.utils.data import Dataset, DataLoader
from losses import TripletLoss, OnlineTripletLoss
from networks import EmbeddingNet, EmbeddingNetL2, ClassificationNet, TripletNet
from torch.cuda.amp import autocast,GradScaler
import torch.nn as nn
import torch.nn.functional as F
import time
from model import make_model
from tqdm import tqdm


# def test_embedding_retrieval(model, dataset, device="cuda"):
#     model.eval()
#     embeddings = []
#     labels = []

#     # 先提取所有图片的 embedding 和标签
#     with torch.no_grad():
#         for i in range(len(dataset)):
#             anc_img, _, _, label = dataset[i]
#             anc_img = anc_img.unsqueeze(0).to(device)  # [1, C, H, W]
#             embedding, _ = model.get_embedding(anc_img), None
#             embeddings.append(embedding.cpu())
#             labels.append(label.item())

#     embeddings = torch.cat(embeddings, dim=0)  # [N, D]
#     labels = torch.tensor(labels)  # [N]

#     correct = 0
#     total = len(dataset)

def test_embedding_retrieval(model, dataloader, val_dataloader, device="cuda"):
    model.eval()
    db_embeddings = []
    db_labels = []
    db_img_paths = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting DB embeddings"):
            anc_img = batch[0].to(device)
            label = batch[3]
            # 获取图片路径（假设 __getitem__ 返回 anc_img, pos_img, neg_img, label, anc_img_path, ...）
            if len(batch) >= 5:
                anc_img_path = batch[4]
                db_img_paths.extend(anc_img_path)
            else:
                db_img_paths.extend([None]*anc_img.size(0))
            embedding = model.get_embedding(anc_img)
            db_embeddings.append(embedding)
            db_labels.extend(label.numpy().tolist())
    db_embeddings = torch.cat(db_embeddings, dim=0)  # [N_db, D]
    db_labels = torch.tensor(db_labels)  # [N_db]

    correct_top1 = 0
    correct_top5 = 0
    correct_top1_percent = 0
    total = len(val_dataloader.dataset)
    top1_percent_k = max(1, int(total * 0.01))

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Querying"):
            anc_img = batch[0].to(device)
            label = batch[3]

            query_embeddings = model.get_embedding(anc_img)  # [B, D]
            query_labels = label.numpy().tolist()                  # [B]

            for i in range(query_embeddings.size(0)):
                query_emb = query_embeddings[i]
                query_label = query_labels[i]
                dists = torch.norm(db_embeddings - query_emb, dim=1)

                # top1
                nn_idx = torch.argmin(dists).item()
                if query_label == db_labels[nn_idx]:
                    correct_top1 += 1

                # top5
                top5_idx = torch.topk(dists, k=5, largest=False).indices
                if any(query_label == db_labels[idx] for idx in top5_idx):
                    correct_top5 += 1

                # top1%
                topk_idx = torch.topk(dists, k=top1_percent_k, largest=False).indices
                if any(query_label == db_labels[idx] for idx in topk_idx):
                    correct_top1_percent += 1
                

    print(f'Top-1 accuracy: {100 * correct_top1 / total:.2f}%')
    print(f'Top-5 accuracy: {100 * correct_top5 / total:.2f}%')
    print(f'Top-1% accuracy: {100 * correct_top1_percent / total:.2f}%')

if __name__ == "__main__":
    # root_dir = "./BI" # Formal test
    root_dir = "./OBC" # Benchmark test
    dataset = Tripletdataset(root_dir, get_transform())
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
        generator=generator
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
    # Load trained model
    num_classes = len(train_dataset.dataset.all_classes)
    print(num_classes)
    model = make_model(num_classes=num_classes).cuda()
    model.load_state_dict(torch.load('./model/OBC_best_model_without_classifier.pth'))


    test_embedding_retrieval(model, dataloader, val_dataloader, device="cuda")
