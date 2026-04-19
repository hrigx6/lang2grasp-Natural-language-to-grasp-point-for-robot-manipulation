# Lang2Grasp

A robot perception pipeline that turns natural language instructions into grasp points using Qwen2-VL, Grounding DINO, and SAM. Type an instruction, get back segmented objects and grasp coordinates.

## models used

- [Qwen2-VL 2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) — vision language model for instruction parsing
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — open vocabulary object detection
- [SAM ViT-B](https://github.com/facebookresearch/segment-anything) — segment anything model

## setup

```bash
conda create -n vla python=3.10 -y
conda activate vla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.49.0 groundingdino-py segment-anything qwen-vl-utils opencv-python pillow bitsandbytes accelerate
```

download weights:
```bash
mkdir weights
wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

login to huggingface:
```bash
hf auth login
```

## usage

```bash
python pipeline.py
image path (enter for default): 
instruction: pick up the cup and place it on the notebook
```

outputs:
- `pipeline_result.png` — original image with segmentation overlay and markers
- `task_space.png` — black background showing only detected objects
- `combined_view.png` — side by side view

## what works

- basic pick and place
- between two reference objects
- multi-step instructions (partial)
- any object DINO can detect by name

## limitations

- synonym understanding fails ("writing instrument" ≠ "pen")
- size/spatial reasoning unreliable ("smallest object")
- ~10 second inference per query — not real time
- requires GPU with 6GB+ VRAM

## hardware

tested on RTX 3050 6GB laptop, Ubuntu 22.04, CUDA 12.4

## roadmap

- [ ] OpenVLA end-to-end inference
- [ ] real-time video stream mode
- [ ] Jetson Orin deployment
- [ ] fine-tuning on custom objects
