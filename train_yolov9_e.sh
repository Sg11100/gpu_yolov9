#!/bin/bash

# 设置GPU训练环境变量
export CUDA_VISIBLE_DEVICES=0

python  train_dual.py \
--workers 8  \
--batch 16 --epochs 100 --img 512 --device 0  --min-items 0 --close-mosaic 15 \
--data /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/pcbdata1/data/data.yaml \
--cfg /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9/models/detect/yolov9-e.yaml \
--hyp /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9/data/hyps/hyp.scratch-high.yaml  \
--name yolov9_e_


