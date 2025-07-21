import torch
import numpy as np
from PIL import Image
import os
from OBI_BI_dataset import Tripletdataset
from OBI_BI_dataset import get_transform
from torch.utils.data import Dataset, DataLoader
from losses import TripletLoss, OnlineTripletLoss
from networks import EmbeddingNet, EmbeddingNetL2, ClassificationNet, TripletNet
from torch.amp import autocast,GradScaler
import torch.nn as nn
import torch.nn.functional as F
import time
from model import make_model
from tqdm import tqdm
from utils import save_network, save_network_without_classifier


def train_model(model, dataloader, optimizer, scheduler, num_epochs=25, device = "cuda"):

    scaler = GradScaler("cuda")
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss = TripletLoss(margin=1.0)
    min_train_loss = float('inf')
       
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        total_loss = 0.0
        for anc_img, pos_img, neg_img, labels in tqdm(dataloader):
            anc_img = anc_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                anc_output_embedding, pos_output_embedding, neg_output_embedding, anc_output_logits = model(anc_img, pos_img, neg_img, labels)

                loss_triplet = triplet_loss(anc_output_embedding, pos_output_embedding, neg_output_embedding)
                loss_cls = criterion(anc_output_logits, labels)

                loss = loss_triplet + loss_cls
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f'Train Loss: {total_loss/len(dataloader):.4f}')
        if total_loss/len(dataloader) < min_train_loss:
            min_train_loss = total_loss/len(dataloader)
            # model.save_state_dict(f'./model/epoch_{epoch}_best_model.pth')
            save_network(model)

    return model

def train_model_without_classifier(model, dataloader, optimizer, scheduler, num_epochs=25, device = "cuda"):

    scaler = GradScaler("cuda")
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    triplet_loss = TripletLoss(margin=1.0)
    min_train_loss = float('inf')
       
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        total_loss = 0.0
        for anc_img, pos_img, neg_img, labels in tqdm(dataloader):
            anc_img = anc_img.to(device)
            pos_img = pos_img.to(device)
            neg_img = neg_img.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                anc_output_embedding, pos_output_embedding, neg_output_embedding, anc_output_logits = model(anc_img, pos_img, neg_img, labels)

                loss_triplet = triplet_loss(anc_output_embedding, pos_output_embedding, neg_output_embedding)
                loss = loss_triplet
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f'Train Loss: {total_loss/len(dataloader):.4f}')
        if total_loss/len(dataloader) < min_train_loss:
            min_train_loss = total_loss/len(dataloader)
            # model.save_state_dict(f'./model/epoch_{epoch}_best_model.pth')
            save_network_without_classifier(model)

    return model

if __name__ == "__main__":
    root_dir = "./OBC" # For benchmark
    # root_dir = "./BI" # For formal testing
    dataset = Tripletdataset(root_dir, get_transform())
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
        generator=generator
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print(f"Train dataset size: {len(train_dataset)}")
    num_classes = len(train_dataset.dataset.all_classes)
    model = make_model(num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # train_model(model, train_dataloader, optimizer, scheduler, num_epochs=10)
    train_model_without_classifier(model, train_dataloader, optimizer, scheduler, num_epochs=10)
