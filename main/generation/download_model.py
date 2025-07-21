import modelscope
from huggingface_hub import snapshot_download

# model_dir = modelscope.snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='./model/', revision='master')

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    cache_dir="./model/stable-diffusion-2-1",
    revision="main"
)