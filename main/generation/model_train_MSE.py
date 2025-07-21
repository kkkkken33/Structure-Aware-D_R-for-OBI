import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from loss import EmbeddingLoss
from judge_model import make_judge_model
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OBI2BIDataset, get_transform
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.amp import autocast, GradScaler

class FullNet(nn.Module):
    def __init__(self, unet_path, text_encoder_path, tokenizer_path, vae_path):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.vae = AutoencoderKL.from_pretrained(vae_path)

    def get_text_emb(self, text_list, device):
        inputs = self.tokenizer(text_list, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids)[0]
        return text_emb
    
    def get_vae_latent(self, img):
        latent = self.vae.encode(img).latent_dist.sample() * 0.18215
        return latent

    def vae_latent_to_img(self, latent):
        img = self.vae.decode(latent / 0.18215).sample
        return img

    def forward(self, x, t, text_emb, noise):
        if noise is None:
            noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, t)
        pred = self.unet(noisy_x, t, encoder_hidden_states=text_emb).sample
        return pred



if __name__ == "__main__":
    debug_file = "./debug.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # judge_model = make_judge_model().to(device)
    # judge_model.load_state_dict(torch.load('./model/judge_model.pth', map_location=device))
    # judge_model.eval()
    # for param in judge_model.parameters():
    #     param.requires_grad = False
    # embedding_loss = EmbeddingLoss(judge_model).to(device)

    # 下载的预训练模型路径
    vae_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/vae"
    unet_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/unet"
    text_encoder_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/text_encoder"
    tokenizer_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/tokenizer"

    # 构造模型
    model = FullNet(unet_path, text_encoder_path, tokenizer_path, vae_path).to(device)
    # 冻结vae和text_encoder
    for param in model.vae.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # 数据集
    dataset = OBI2BIDataset(root_dir="./dataset", json_file="./dataset/dataset.json", transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    # 训练代码
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.train()
    min_loss = float('inf')
    for epoch in range(5):
        print(f"Epoch {epoch + 1}/{5}")
        total_loss = 0.0
        for obc_img, bi_img, obc_text in tqdm(dataloader):
            # with open(debug_file, 'a') as f:
            #     f.write(f"obc_img: {obc_img.min().item()}, {obc_img.max().item()}, {obc_img.mean().item()}\n")
            #     f.write(f"bi_img: {bi_img.min().item()}, {bi_img.max().item()}, {bi_img.mean().item()}\n")

            obc_img = obc_img.to(device)
            bi_img = bi_img.to(device)
            obc_latent = model.get_vae_latent(obc_img).to(device)
            # bi_latent = model.get_vae_latent(bi_img).to(device)
            text_emb = model.get_text_emb(obc_text, device)

            # 采样timestep
            batch_size = obc_latent.size(0)
            t = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(obc_latent)

            # 更新参数
            optimizer.zero_grad()

            # 前向
            with autocast(device_type="cuda"):
                pred_noise = model(obc_latent, t, text_emb, noise=noise)

                # 噪声预测损失（MSE）
                noise_pred_loss = torch.nn.functional.mse_loss(pred_noise, noise)


                # 总损失（可调整权重）
                loss = noise_pred_loss

                # with open(debug_file, 'a') as f:
                #     f.write(f"pred_noise: {pred_noise.min().item()}, {pred_noise.max().item()}\n")
                #     f.write(f"obc_latent: {obc_latent.min().item()}, {obc_latent.max().item()}\n")
                #     f.write(f"loss: {loss.item()}\n")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # checkpoint

        # 保存最优模型和存档
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, f'./model/epoch_{epoch}_checkpoint.pth')

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), './model/obi2bi_model.pth')
        
        torch.cuda.empty_cache()