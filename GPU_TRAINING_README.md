# YOLOv9 GPU训练配置说明

## 修改内容总结

已成功将YOLOv9从NPU训练转换为GPU训练，主要修改包括：

### 1. 训练脚本修改
- **文件**: `train_yolov9_e.sh`
- **修改**: 将 `export ASCEND_LAUNCH_BLOCKING=1` 改为 `export CUDA_VISIBLE_DEVICES=0`
- **新增**: `train_yolov9_e_gpu.sh` (带GPU状态检查的完整版本)

### 2. 训练代码修改
- **文件**: `train_dual.py`
- **移除**: 所有NPU相关导入和配置
  - `import torch_npu`
  - `from torch_npu.npu import amp`
  - `from torch_npu.contrib import transfer_to_npu`
  - `torch_npu.npu.set_compile_mode(jit_compile=False)`
- **替换**: NPU的GradScaler为CUDA版本
  - `amp.GradScaler(init_scale=2)` → `torch.cuda.amp.GradScaler()`
- **注释**: 所有NPU性能分析相关代码

### 3. 模型文件修改
- **文件**: `models/common.py`
- **移除**: NPU相关导入 `import torch_npu`
- **替换**: NPU专用激活函数为标准版本
  - 注释掉 `torch_npu.contrib.module.SiLU()` 相关代码
  - 使用标准 `nn.SiLU()`
- **移除**: NPU专用Linear层 `NpuLinear`
- **替换**: 所有 `NpuLinear` 使用为标准 `nn.Linear`

### 4. 工具函数修改
- **文件**: `utils/torch_utils.py`
- **移除**: NPU相关导入
  - `import torch_npu`
  - `from torch_npu.npu import amp`
  - `import apex`
- **移除**: NPU专用Linear层和优化器
- **替换**: 所有NPU优化器为标准PyTorch版本
  - `torch_npu.optim.NpuFusedSGD` → `torch.optim.SGD`
  - `apex.optimizers.NpuFusedAdam` → `torch.optim.Adam`
  - 等等

### 5. 通用函数修改
- **文件**: `utils/general.py`
- **移除**: NPU相关导入 `import torch_npu`
- **替换**: NPU专用NMS为标准版本
  - 移除 `torch_npu.npu_nms_v4` 条件分支
  - 统一使用 `torchvision.ops.nms`

### 6. 验证脚本修改
- **文件**: `val_dual.py`
- **移除**: NPU相关导入
  - `import torch_npu`
  - `from torch_npu.contrib import transfer_to_npu`

### 7. 检测脚本修改
- **文件**: `detect.py`
- **注释**: NPU专用函数
  - `torch_npu.contrib.function.npu_fast_condition_index_put`

### 8. 环境检查
- **新增**: `test_gpu_env.py` GPU环境测试脚本

## 使用方法

### 方法1：使用修改后的原始脚本
```bash
cd /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9
./train_yolov9_e.sh
```

### 方法2：使用新的GPU训练脚本（推荐）
```bash
cd /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9
./train_yolov9_e_gpu.sh
```

### 方法3：直接使用Python命令
```bash
cd /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9
python train_dual.py \
--workers 8 \
--batch 16 \
--epochs 100 \
--img 512 \
--device 0 \
--min-items 0 \
--close-mosaic 15 \
--noval \
--data /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/pcbdata1/data/data.yaml \
--cfg /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9/models/detect/yolov9-e.yaml \
--hyp /root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/yolov9/data/hyps/hyp.scratch-high.yaml \
--name yolov9_e_gpu
```

## 环境要求验证

运行以下命令验证环境是否正确配置：
```bash
python test_gpu_env.py
```

## 训练参数说明

- `--device 0`: 使用第一个GPU (cuda:0)
- `--batch 16`: 批处理大小，可根据显存调整
- `--epochs 100`: 训练轮数
- `--img 512`: 输入图像大小
- `--workers 8`: 数据加载进程数
- `--name yolov9_e_gpu`: 实验名称，结果保存在 `runs/train/yolov9_e_gpu*` 文件夹

## 注意事项

1. **显存使用**: 当前批处理大小为16，在A800 80GB显存下应该足够。如果遇到显存不足，可以减小batch size
2. **多GPU训练**: 如果需要多GPU训练，修改 `--device 0,1,2,3` 并相应调整batch size
3. **训练数据**: 确保数据路径正确，当前指向 `/root/autodl-tmp/BLBDGCD_huawei2024/yolov9_npu_train/pcbdata1/data/data.yaml`

## 常见问题

### Q: 训练过程中显存不足怎么办？
A: 减小batch size，例如从16改为8或4

### Q: 如何使用多GPU训练？
A: 修改 `--device 0,1` 并确保batch size能被GPU数量整除

### Q: 训练结果保存在哪里？
A: 保存在 `runs/train/yolov9_e_gpu*` 文件夹中

### Q: 如何恢复训练？
A: 添加 `--resume` 参数指向之前的checkpoint

## 环境信息

- **PyTorch版本**: 2.1.2+cu121
- **CUDA版本**: 12.1
- **GPU**: NVIDIA A800 80GB PCIe
- **Python版本**: 3.10.8

## 修改文件清单

以下文件已被修改以支持GPU训练：

1. `train_yolov9_e.sh` - 训练脚本环境变量
2. `train_dual.py` - 主训练代码NPU→GPU
3. `models/common.py` - 模型组件NPU→GPU
4. `utils/torch_utils.py` - PyTorch工具函数NPU→GPU
5. `utils/general.py` - 通用函数NPU→GPU
6. `val_dual.py` - 验证脚本NPU→GPU
7. `detect.py` - 检测脚本NPU相关注释

**新增文件**:
- `train_yolov9_e_gpu.sh` - GPU训练脚本（推荐）
- `test_gpu_env.py` - GPU环境测试脚本
- `GPU_TRAINING_README.md` - 本说明文档

修改完成！现在可以使用GPU进行YOLOv9训练了。
