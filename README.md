# Project Title

## 📝 Project Overview

This is the final project of DSAA-2012. The work aims to provide a novel pipeline for oracle bone inscription decipher, including generation for bronze images from given oracle bone images and retrieval for visually similar characters.

Our key contributions are:
- A novel application of conditional diffusion model guided by a VLLM to generate intermediate-script represen-
tations (BI) from ancient oracle bone inputs.
- A hybrid pipeline that combines image generation with visual retrieval to infer semantic
meaning.
- A curated dataset of OBC–BI–semantic triplets to support supervised training and evaluation.

## 📁 Download data
You can download open oracle bone and bronze inscription dataset from EVOBC. https://github.com/RomanticGodVAN/character-Evolution-Dataset

Reform the original dataset to this structure.

```bash
dataset/
├── train/
│   ├── OBC/
│   │   ├── cls1/
│   │   │   ├── img1.png
│   │   │   ├── img2.png
│   │   │   └── ...
│   │   ├── cls2/
│   │   │   ├── img1.png
│   │   │   └── ...
│   │   └── ...
│   ├── BI/
│   │   ├── cls1/
│   │   │   ├── img1.png
│   │   │   ├── img2.png
│   │   │   └── ...
│   │   ├── cls2/
│   │   │   ├── img1.png
│   │   │   └── ...
│   │   └── ...
├── test/
│   ├── OBC/
│   │   └── ...
│   ├── BI/
│   │   └── ...
```
## 🚀 Usage
### Recommended
- Python >= 3.8
- (Optional) Use a virtual environment (e.g., venv, conda)
### Install dependencies
```bash
pip install -r requirement.txt
```
### Use generation model
```bash
cd generation
pip install -r requirement.txt
python download_model.py
python model_train_MSE.py
python inference.py
```
### Use retrieval model
```bash
cd retrieval
pip install -r requirement.txt
python train.py
python test.py
```

