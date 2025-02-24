# Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach

This repository is a PyTorch implementation of our paper: Enhancing Multi-Scale Fabric Defect Detection with MCF-Net: A Context Fusion Approach.

## DataSet
The FD6052 dataset of this study is open-sourced at https://www.kaggle.com/datasets/wyyt1202/fd6052.

TianChi dataset can be downloaded from https://tianchi.aliyun.com/dataset/79336.

DAGM2007 dataset can be downloaded from https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection.

## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).
  
```bash
git clone https://github.com/wyyt1202/MCF-Net  # clone
cd MCF-Net
pip install -r requirements.txt  # install
```
</details>

<details open>
<summary>Train</summary>
  
Training with `train.py` and saving results to `runs/train`.

Single-GPU

`python train.py --model MCF-Net.yaml --epochs 5 --img 640 --batch 32 --data ../datasets/fabric.yaml`

Multi-GPU DDP

`python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --model MCF-Net.yaml --epochs 5 --img 640 --device 0,1,2,3 --data ../datasets/fabric.yaml`
</details>

<details open>
<summary>Val</summary>

`val.py` runs on a validation set and saving results to `runs/val`.
```
python val.py --weights MCF-Net.pt --img 640 --batch 32 --data ../datasets/fabric.yaml
```
</details>

<details open>
<summary>Inference</summary>

`detect.py` runs inference on a variety of sources and saving results to `runs/detect`.
```
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
</details>

