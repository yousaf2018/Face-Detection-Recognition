# Facial Recognition Project

## Overview

This project implements facial recognition. It allows you to perform face detection and recognition within images or video streams.

## Getting Started

Follow the instructions below to set up and run the facial recognition project.

Download microsoft build tool, select desktop development with C++ during installation
https://drive.google.com/file/d/1xEW4pM_orE9qHMureu0krFsRvv5k_qxZ/view


### Clone the Repository
```bash
git clone https://github.com/A-Jabbar/face-recognition.git
cd face-recognition
pip install -r requirements.txt
```

### Training facial model

```bash
python3 source_code/training/train.py --img 640 --batch 16 --epochs 10 --data source_code/training/data/custom.yaml --weights source_code/training/yolov5s.pt --nosave --cache
```

## Inference on Trained Model

## Project Structure
- `source_code/training`: Contains training-related code and output.
- `source_code/testing`: Contains testing-related code and models.

## After above steps run below cammand for inferernce 
```bash
python3 inference.py
```