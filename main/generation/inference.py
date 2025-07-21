import torch
from torchvision.utils import save_image
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from model_train_MSE import FullNet
from PIL import Image
from torchvision import transforms
import json
import os

# 路径配置
vae_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/vae"
unet_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/unet"
text_encoder_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/text_encoder"
tokenizer_path = "./model/stable-diffusion-2-1/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6/tokenizer"
model_ckpt = "./model/epoch_4_checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = FullNet(unet_path, text_encoder_path, tokenizer_path, vae_path).to(device)
checkpoint = torch.load('./model/epoch_0_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理函数
def generate_image(obc_img_path, obc_text, output_path, num_inference_steps=50):
    # 预处理图片
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        lambda x: x * 2.0 - 1.0,  # [-1, 1]
    ])
    img = Image.open(obc_img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    # 文本编码
    text_emb = model.get_text_emb([obc_text], device)

    # VAE编码
    latent = model.get_vae_latent(img)

    # DDPM采样
    scheduler = model.noise_scheduler
    scheduler.set_timesteps(num_inference_steps)
    noise = torch.randn_like(latent)
    x = latent + noise  # 初始加噪

    for t in scheduler.timesteps:
        with torch.no_grad():
            pred_noise = model.unet(x, t, encoder_hidden_states=text_emb).sample
        x = scheduler.step(pred_noise, t, x).prev_sample

    # VAE解码
    with torch.no_grad():
        gen_img = model.vae_latent_to_img(x)
    gen_img = (gen_img.clamp(-1, 1) + 1) / 2  # [0,1]
    save_image(gen_img, output_path)
    print(f"Saved generated image to {output_path}")

if __name__ == "__main__":
    # 读取 dataset_test.json
    json_path = "./dataset/dataset_test.json"
    input_dir = "./dataset"
    output_dir = "./generated_results"
    os.makedirs(output_dir, exist_ok=True)
    i = 100
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            i += 1
            if i > 500:
                break
            obc_img_path = data["obc_img"]
            obc_text = data["obc_text_description"]
            img_path = os.path.join(input_dir, obc_img_path)
            img_name = os.path.basename(obc_img_path)
            output_path = os.path.join(output_dir, img_name)
            generate_image(img_path, obc_text, output_path)